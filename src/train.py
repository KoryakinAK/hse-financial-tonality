from collections import defaultdict
import pandas as pd
import sklearn
from sklearn.model_selection import KFold, TimeSeriesSplit
from sklearn.pipeline import Pipeline


metric_func_dict = {
    "acc": lambda y_true, y_pred: (y_true == y_pred).mean(),
    "f1": lambda y_true, y_pred: sklearn.metrics.f1_score(
        y_true,
        y_pred,
        average="macro",
    ),
}


def train_and_evaluate(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float]:
    pipeline.fit(X_train, y_train)
    pred_test = pipeline.predict(X_test)

    return {metric_name: func(y_test, pred_test) for metric_name, func in metric_func_dict.items()}


def train_and_cross_validate(
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 3,
    split_type: str = "time_series",
) -> dict[str, list[float]]:
    split_type_dict = {
        "kfold": KFold,
        "time_series": TimeSeriesSplit,
    }

    X = X.copy().reset_index(drop=True)
    y = y.copy().reset_index(drop=True)

    metrics_dict = defaultdict(list)

    splitter = split_type_dict[split_type](
        n_splits=n_splits,
    )
    for i, (train_index, test_index) in enumerate(splitter.split(X)):
        pipeline.fit(X.loc[train_index], y[train_index])
        pred_test = pipeline.predict(X.loc[test_index])

        for metric_name, func in metric_func_dict.items():
            metrics_dict[metric_name].append(func(y[test_index], pred_test))

    return metrics_dict
