# Take raw csv data, return processed forecasting data

import pandas as pd
import os
import sklearn
from tsai.basics import (
    ndarray,
    TSShrinkDataFrame,
    TSDropDuplicates,
    TSAddMissingTimestamps,
    TSFillMissing,
    get_forecasting_splits,
    TSStandardScaler,
    prepare_forecasting_data,
    mkdir,
    save_object,
    np,
)


def process_data(path_to_raw_data) -> ndarray:
    """
    Processes raw CSV data for forecasting by performing a series of data transformation steps.
    This function also saves two preprocessed pipelines in the ~/data/processed/ directory.

    Parameters:
    path_to_raw_data (str): Path to the directory containing the raw CSV data file.

    Returns:
    tuple: A tuple containing two elements:
           - X (ndarray): The feature matrix for the forecasting model.
           - y (ndarray): The target variable for the forecasting model.
    """
    df = pd.read_csv(os.path.join(path_to_raw_data, "cex4WindDataInterpolated.csv"))

    # convert time to datetime
    df["t"] = pd.to_datetime(df["t"])

    # TODO add it to logs (?)
    # df.head()

    datetime_col = "t"
    freq = "24h"
    columns = df.columns[1:]
    method = "ffill"
    value = 0

    # pipeline
    preproc_pipe = sklearn.pipeline.Pipeline(
        [
            ("shrinker", TSShrinkDataFrame()),  # shrink dataframe memory usage
            ("drop_duplicates", TSDropDuplicates(datetime_col=datetime_col)),  # drop duplicate rows (if any)
            (
                "add_mts",
                TSAddMissingTimestamps(datetime_col=datetime_col, freq=freq),
            ),  # add missing timestamps (if any)
            (
                "fill_missing",
                TSFillMissing(columns=columns, method=method, value=value),
            ),  # fill missing data (1st ffill. 2nd value=0)
        ],
        verbose=True,
    )
    mkdir("data", exist_ok=True, parents=True)

    # save_object(preproc_pipe, 'data/preproc_pipe.pkl')
    # preproc_pipe = load_object('data/processed/preproc_pipe.pkl')

    df = preproc_pipe.fit_transform(df)

    # TODO Add this to logs (?)
    # print(df)

    fcst_history = 104  # # steps in the past
    fcst_horizon = 60  # # steps in the future
    valid_size = 0.1  # int or float indicating the size of the training set
    test_size = 0.2  # int or float indicating the size of the test set

    # TODO Move output to figures (param show_plot)
    splits = get_forecasting_splits(
        df,
        fcst_history=fcst_history,
        fcst_horizon=fcst_horizon,
        datetime_col=datetime_col,
        valid_size=valid_size,
        test_size=test_size,
        show_plot=False,
    )

    # TODO Add this to logs
    # splits

    columns = df.columns[1:]
    train_split = splits[0]

    # pipeline
    exp_pipe = sklearn.pipeline.Pipeline(
        [
            ("scaler", TSStandardScaler(columns=columns)),  # standardize data using train_split
        ],
        verbose=True,
    )

    # save_object(exp_pipe, 'data/exp_pipe.pkl')
    # exp_pipe = load_object('data/processed/exp_pipe.pkl')

    df_scaled = exp_pipe.fit_transform(df, scaler__idxs=train_split)

    # TODO Add this to logs (?)
    # print(df_scaled)

    # TODO check it -> different than notebook
    x_vars = df_scaled.columns[1:]
    y_vars = df_scaled.columns[1:]

    X, y = prepare_forecasting_data(
        df, fcst_history=fcst_history, fcst_horizon=fcst_horizon, x_vars=x_vars, y_vars=y_vars
    )

    return X, y, preproc_pipe, exp_pipe, splits


if __name__ == "__main__":
    PATH_RAW = "data/raw"
    PATH_PROCESSED = "data/processed"

    X, y, preproc_pipe, exp_pipe, splits = process_data(PATH_RAW)

    # TODO Find better solution -> it can take a while (VSCode issue?)
    np.savez(os.path.join(PATH_PROCESSED, "processed.npz"), array1=X, array2=y)
    save_object(preproc_pipe, os.path.join(PATH_PROCESSED, "preproc_pipe.pkl"))
    save_object(exp_pipe, os.path.join(PATH_PROCESSED, "exp_pipe.pkl"))
    save_object(splits, os.path.join(PATH_PROCESSED, "splits.pkl"))
