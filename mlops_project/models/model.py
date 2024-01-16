from tsai.basics import TSForecaster, load_object, mse, mae, np, os

# By using tsai module the model can be reduced just to the architecture definition.


class Forecaster:
    """
    A forecasting model class using the tsai library, focused on time series data.

    Attributes:
    arch_config (dict): Configuration parameters for the model architecture.

    Methods:
    _get_data(path): Loads preprocessed data, pipelines, and splits.
    train_model(): Trains the forecasting model using the TSForecaster from tsai.
                   The model is trained and saved to models directory.
    """

    def __init__(self) -> None:
        self.arch_config = dict(
            n_layers=3,  # number of encoder layers
            n_heads=4,  # number of heads
            d_model=16,  # dimension of model
            d_ff=128,  # dimension of fully connected network
            attn_dropout=0.0,  # dropout applied to the attention weights
            dropout=0.3,  # dropout applied to all linear layers in the encoder except q,k&v projections
            patch_len=24,  # length of the patch applied to the time series to create patches
            stride=2,  # stride used when creating patches
            padding_patch=True,  # padding_patch
        )

    def get_data(self, path):
        feature_target_data = np.load(os.path.join(path, "processed.npz"))
        X = feature_target_data["array1"]
        y = feature_target_data["array2"]

        preproc_pipe = load_object(os.path.join(path, "preproc_pipe.pkl"))
        exp_pipe = load_object(os.path.join(path, "exp_pipe.pkl"))
        splits = load_object(os.path.join(path, "splits.pkl"))

        return X, y, preproc_pipe, exp_pipe, splits

    def train_model(self):
        PATH_PROCESSED = "data/processed"

        X, y, preproc_pipe, exp_pipe, splits = self.get_data(PATH_PROCESSED)

        learn = TSForecaster(
            X,
            y,
            splits=splits,
            batch_size=16,
            path="models",
            pipelines=[preproc_pipe, exp_pipe],
            arch="PatchTST",
            arch_config=self.arch_config,
            metrics=[mse, mae],
        )

        n_epochs = 20
        lr_max = 0.0025
        learn.fit_one_cycle(n_epochs, lr_max=lr_max)

        # save model
        learn.export("patchTST.pt")

        # TODO make a plot
