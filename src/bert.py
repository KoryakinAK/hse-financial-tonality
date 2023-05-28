from collections import defaultdict
from copy import deepcopy
from typing import Any, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, TimeSeriesSplit, train_test_split
from tqdm import tqdm
import torch
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import transformers
from transformers import logging

from happytransformer import HappyTextClassification

from src import engine, train

logging.set_verbosity_error()


bert_features_cache = {}


def split_into_chunks(l: list[Any], n: int):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(l), n):
        yield l[i : i + n]


def tokenizer_input_to_device(
    tokenizer_input: dict[str, torch.Tensor],
    device: Union[str, torch.device],
) -> dict[str, torch.Tensor]:
    return {attr: val.to(device) for attr, val in tokenizer_input.items()}


def tokenizer_input_get_idx(
    tokenizer_input: dict[str, torch.Tensor],
    idx: int,
) -> dict[str, torch.Tensor]:
    return {attr: val[idx] for attr, val in tokenizer_input.items()}


class BertVectorizer(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        bert_name: str = "bert-base-uncased",
        pad_max_length: int = 100,
        chunk_size: int = 100,
        use_cache: bool = True,
    ):
        """Sklearn трансформер для создания эмбеддингов слов через Bert

        Parameters
        ----------
        bert_name: str
            Название модели с huggingface
        pad_max_length: int
            Макс колво токенов
        chunk_size : int, optional
            Размер чанка, на котором идет предсказание Bert, by default 10000
        use_cache: bool
            Кэш для инференса берта
        """

        self.bert_name = bert_name
        self.pad_max_length = pad_max_length
        self.chunk_size = chunk_size
        self.use_cache = use_cache
        self.bert_embedding_size = 768
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(bert_name)
        self.bert_model = transformers.AutoModel.from_pretrained(bert_name)

    def fit(self, X, y=None):
        return self

    def get_bert_features(self, texts: list[str]) -> np.ndarray:
        if self.use_cache:
            cache_key = tuple(texts + [self.bert_name])

            output = bert_features_cache.get(cache_key, None)
            if output is not None:
                return output

        encoded_input = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.pad_max_length,
        )
        encoded_input = tokenizer_input_to_device(encoded_input, "cuda")
        self.bert_model.to("cuda")
        output = self.bert_model(**encoded_input)
        # maybe should use pooler_output
        output = output[0][:, 0, :].detach().cpu().numpy()

        if self.use_cache:
            bert_features_cache[cache_key] = output

        return output

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        X_list_chunks = split_into_chunks(X.tolist(), self.chunk_size)
        num_chunks = (len(X) - 1) // self.chunk_size + 1

        outputs = []

        for chunk in tqdm(X_list_chunks, total=num_chunks, desc=f"Bert inference on {X.shape[0]} texts"):
            output = self.get_bert_features(chunk)
            outputs.append(output)

        return np.concatenate(outputs, axis=0)


class BertHappyPredictor(TransformerMixin, BaseEstimator):
    def __init__(
        self,
        *,
        model_type: str = "DISTILBERT",
        model_name: str = "distilbert-base-uncased-finetuned-sst-2-english",
    ):
        self.model_type = model_type
        self.model_name = model_name
        self.happy_bert_model = HappyTextClassification(model_type=model_type, model_name=model_name)

    def fit(self, X, y=None):
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        output = np.full(
            X.shape[0],
            "",
            dtype="object",
        )
        for i, sentence in enumerate(X.values.flatten()):
            pred = self.happy_bert_model.classify_text(sentence)
            output[i] = pred.label.lower()

        return output


import shutil
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.tensorboard.writer import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR, ChainedScheduler
from torch.utils.data import DataLoader


class BertDataset(Dataset):
    def __init__(
        self,
        texts,
        labels,
        pad_max_length=100,
        bert_name="distilbert-base-uncased",
        device="cuda",
    ):
        self.bert_name = "distilbert-base-uncased"
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(bert_name)
        self.pad_max_length = pad_max_length

        self.encoded_inputs = self.tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.pad_max_length,
        )
        self.labels = labels
        self.label_to_id = {label: i for i, label in enumerate(sorted(set(labels)))}
        self.id_to_label = {i: label for label, i in self.label_to_id.items()}
        self.device = device

    def __getitem__(self, idx):
        encoded_input = tokenizer_input_get_idx(self.encoded_inputs, idx)
        encoded_input = tokenizer_input_to_device(encoded_input, self.device)
        label_i = self.label_to_id[self.labels.iloc[idx]]
        return encoded_input, torch.tensor(label_i, device=self.device)

    def __len__(self):
        return len(self.labels)


class BertFineTuned(nn.Module):
    def __init__(
        self,
        num_classes=3,
        bert_name="distilbert-base-uncased",
        use_softmax_in_forward=False,
    ):
        super().__init__()
        self.bert_name = "distilbert-base-uncased"
        self.bert_model = transformers.AutoModel.from_pretrained(bert_name)

        layers = [
            nn.Linear(self.bert_model.config.dim, 200),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(200, 100),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(100, num_classes),
        ]

        if use_softmax_in_forward:
            layers.append(nn.Softmax(dim=1))
        self.use_softmax_in_forward = use_softmax_in_forward

        self.head = nn.Sequential(*layers)

        for param in self.bert_model.parameters():
            param.requires_grad = False

    def forward(self, encoded_input):
        x = self.bert_model(**encoded_input)
        x = self.head(x[0][:, 0, :])
        return x

    def predict(self, encoded_input):
        x = self.forward(encoded_input)
        if not self.use_softmax_in_forward:
            x = x.softmax(dim=1)
        x = x.argmax()
        return x


class BertTrainer(engine.Trainer):
    def compute_loss_on_batch(self, batch) -> torch.Tensor:
        # Считает лосс и метрики
        x, y = batch

        logits = self.model.forward(x)

        return self.criterion(logits, y)


def train_bert_and_cross_validate(
    model: BertFineTuned,
    X: pd.DataFrame,
    y: pd.Series,
    criterion,
    lr: float = 0.005,
    milestones: list[int] = [],
    device: str = "cuda",
    epochs: int = 1,
    n_splits: int = 3,
    split_type: str = "time_series",
    num_workers: int = 0,
    verbose: bool = True,
    dont_reuse_folds: bool = False,
) -> dict[str, list[float]]:
    split_type_dict = {
        "kfold": KFold,
        "time_series": TimeSeriesSplit,
    }

    # in time_series scenaria we dont reset model
    if split_type == "kfold":
        base_model = deepcopy(model)

    X = X.copy().reset_index(drop=True)
    y = y.copy().reset_index(drop=True)

    metrics_dict = defaultdict(list)

    splitter = split_type_dict[split_type](
        n_splits=n_splits,
    )
    splits = list(splitter.split(X))

    for i, (train_index, test_index) in enumerate(splits):
        if split_type == "kfold":
            model = deepcopy(base_model)

        # remove elements from prev_train_index
        if dont_reuse_folds and split_type == "time_series" and i > 0:
            prev_train_index, _ = splits[i - 1]
            train_index = train_index[~np.isin(train_index, prev_train_index)]

        train_index, val_index = train_test_split(train_index, test_size=0.1, random_state=69)
        texts_train = X.loc[train_index]["text"].tolist()
        train_dataset = BertDataset(texts_train, y[train_index], device=device)

        texts_val = X.loc[val_index]["text"].tolist()
        val_dataset = BertDataset(texts_val, y[val_index], device=device)
        # hardcode...
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

        trainer = BertTrainer(
            model,
            optimizer,
            criterion,
            train_dataset,
            val_dataset,
            scheduler=scheduler,
            num_workers=num_workers,
        )
        trainer.train(epochs)
        model = torch.load(trainer.best_model_fp)

        texts_test = X.loc[test_index]["text"].tolist()
        test_dataset = BertDataset(texts_test, y[test_index], device=device)
        test_dataloader = DataLoader(test_dataset)

        model.eval()
        pred_test = [model.predict(x).item() for x, _ in test_dataloader]
        pred_test = np.array([train_dataset.id_to_label[pred] for pred in pred_test])

        for metric_name, func in train.metric_func_dict.items():
            metric_value = func(y[test_index], pred_test)
            metrics_dict[metric_name].append(metric_value)
            if verbose:
                print(f"{metric_name}: {metric_value:.2f}")

    return metrics_dict
