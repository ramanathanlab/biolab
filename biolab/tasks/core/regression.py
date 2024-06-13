from sklearn.svm import SVR
from datasets import Dataset

from biolab.api.metric import Metric


def sklearn_svr(
    task_dset: Dataset, input_col: str, target_col: str, metrics: list[Metric]
):
    """Train a Support Vector Regressor (SVR) using the embeddings and target values.

    Parameters
    ----------
    task_dset : Dataset
        The dataset containing the input features and target labels
    input_col : str
        The name of the column containing the input features
    target_col : str
        The name of the column containing the target labels

    Returns
    -------
    Tuple
        The trained SVR regressor, train mse, and test mse
    """
    # set dset to numpy for this function, we can return it to original later
    dset_format = task_dset.format
    task_dset.set_format("numpy")

    # Split the data into train and test sets
    # TODO: take the seed to somewhere more central this is in the weeds
    svr_dset = task_dset.train_test_split(test_size=0.2, seed=42)

    X_train = svr_dset["train"][input_col]
    y_train = svr_dset["train"][target_col]
    X_test = svr_dset["test"][input_col]
    y_test = svr_dset["test"][target_col]

    # Train the SVR regressor
    regressor = SVR()
    regressor.fit(X_train, y_train)

    # Calculate metrics
    y_train_pred = regressor.predict(X_train)  # noqa: F841
    y_test_pred = regressor.predict(X_test)

    for metric in metrics:
        metric.evaluate(input=y_test_pred, labels=y_test)

    # return dset to original format
    task_dset.set_format(**dset_format)

    return metrics
