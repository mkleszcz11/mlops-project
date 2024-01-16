from tsai.inference import load_learner
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tsai.basics import sys, Path, to_np, pd, plot_forecast, ndarray

from models.model import Forecaster

sys.path.append(str(Path(__file__).parents[1]))


def test_split(X, y, splits) -> ndarray:
    learn = load_learner("models/patchTST.pt")
    y_test_preds, *_ = learn.get_X_preds(X[splits[2]])
    y_test_preds = to_np(y_test_preds)
    print(f"y_test_preds.shape: {y_test_preds.shape}")

    y_test = y[splits[2]]
    results_df = pd.DataFrame(columns=["mse", "mae"])
    results_df.loc["test", "mse"] = mean_squared_error(y_test.flatten(), y_test_preds.flatten())
    results_df.loc["test", "mae"] = mean_absolute_error(y_test.flatten(), y_test_preds.flatten())

    # TODO add to logs
    # results_df

    return y_test_preds


def visualize_predictions(splits, y_test_preds) -> None:
    X_test = X[splits[2]]
    y_test = y[splits[2]]
    # TODO plot it to the 'figures; dir
    plot_forecast(X_test, y_test, y_test_preds, sel_vars=True)


if __name__ == "__main__":
    model_to_vis = Forecaster()
    path_to_processed_data = "data/processed"
    X, y, _, _, splits = model_to_vis.get_data(path_to_processed_data)

    y_test_preds = test_split(X, y, splits)
    visualize_predictions(splits, y_test_preds)
