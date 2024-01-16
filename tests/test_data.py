import os

import pytest
import numpy as np

from tests import _PATH_DATA


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_size():
    feature_target_data = np.load(os.path.join(_PATH_DATA, "processed/processed.npz"))
    X = feature_target_data["array1"]
    y = feature_target_data["array2"]

    # Check if X and y are not empty
    assert X.size > 0, "X array is empty"
    assert y.size > 0, "y array is empty"

    # Check the shape of X and y
    assert len(X.shape) == 3, "X array should be 3D"
    assert len(y.shape) == 3, "y array should be 3D"

    # Check the first dimension is the same for X and y
    assert X.shape[0] == y.shape[0], "Mismatch in the number of samples between X and y"
    assert X.shape[1] == y.shape[1], "Mismatch in the number of features between X and y"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_data_types():
    feature_target_data = np.load(os.path.join(_PATH_DATA, "processed/processed.npz"))
    X = feature_target_data["array1"]
    y = feature_target_data["array2"]

    # Check data types
    assert X.dtype == np.float64 or X.dtype == np.float32, "X array should have floating point type"
    assert y.dtype == np.float64 or y.dtype == np.float32, "y array should have floating point type"


@pytest.mark.skipif(not os.path.exists(_PATH_DATA), reason="Data files not found")
def test_statistical_properties():
    feature_target_data = np.load(os.path.join(_PATH_DATA, "processed/processed.npz"))
    X = feature_target_data["array1"]
    y = feature_target_data["array2"]

    # Basic statistical tests
    assert np.all(np.isfinite(X)), "X contains infinite or NaN values"
    assert np.all(np.isfinite(y)), "y contains infinite or NaN values"
