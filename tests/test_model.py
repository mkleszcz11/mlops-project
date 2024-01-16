import os
import pytest

from mlops_project.models.model import Forecaster

from tests import _PATH_DATA


class TestForecaster:
    @pytest.fixture
    def forecaster(self):
        return Forecaster()

    def test_model_initialization(self, forecaster):
        assert isinstance(forecaster, Forecaster), "Forecaster instance is not created properly"
        assert "n_layers" in forecaster.arch_config, "Architecture config missing 'n_layers'"

    def test_data_loading(self, forecaster):
        PATH_PROCESSED = os.path.join(_PATH_DATA, "processed")
        X, y, _, _, splits = forecaster.get_data(PATH_PROCESSED)
        assert X is not None and y is not None, "Data not loaded correctly"
        assert len(splits) == 3, "Incorrect number of splits"

    def test_training_process(self, forecaster):
        # This test might take longer, consider running with a smaller dataset or fewer epochs
        try:
            forecaster.train_model()
        except Exception as e:
            pytest.fail(f"Training failed with exception: {e}")

    def test_model_saving(self, forecaster):
        model_path = "models/patchTST.pt"
        if os.path.exists(model_path):
            os.remove(model_path)
        # TODO also run for 1/2 epoch
        forecaster.train_model()
        assert os.path.exists(model_path), "Model file not saved"
