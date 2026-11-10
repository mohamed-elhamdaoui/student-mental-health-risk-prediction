import numpy as np
import pytest

from src import config
from src.preprocessing import build_preprocessor, load_data, split_data


@pytest.fixture(scope="module")
def split():
    return split_data(load_data())


def test_split_sizes_and_disjoint(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 1600 and len(X_test) == 400
    assert set(X_train.index).isdisjoint(set(X_test.index))


def test_no_id_or_target_in_features(split):
    X_train, X_test, _, _ = split
    for X in (X_train, X_test):
        assert config.ID_COL not in X.columns
        assert config.TARGET not in X.columns


def test_stratification(split):
    _, _, y_train, y_test = split
    diff = (y_train.value_counts(normalize=True) - y_test.value_counts(normalize=True)).abs()
    assert diff.max() < 0.02


@pytest.mark.parametrize("scale", [True, False])
def test_preprocessor_output_has_no_nan(split, scale):
    X_train, X_test, _, _ = split
    prep = build_preprocessor(scale).fit(X_train)
    out = prep.transform(X_test)
    assert not np.isnan(out).any()
    assert out.shape[1] == 10 + 2 + 2 + 3  # num + ord + bin + one-hot Gender


def test_imputer_statistics_come_from_train_only(split):
    X_train, _, _, _ = split
    prep = build_preprocessor(False).fit(X_train)
    stats = prep.named_transformers_["num"]["imputer"].statistics_
    np.testing.assert_allclose(stats, X_train[config.NUMERIC_FEATURES].median().to_numpy())