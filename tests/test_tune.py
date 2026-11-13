import pandas as pd

from src.tune import select_model


def _df(rows):
    return pd.DataFrame(rows, columns=["model", "f1_cv_mean", "f1_cv_std", "gap_f1_train_minus_cv"]).set_index("model")


def test_prefers_simpler_model_when_scores_are_close():
    results = _df([
        ("LogisticRegression", 0.750, 0.020, 0.01),
        ("RandomForest", 0.755, 0.020, 0.10),
        ("HistGradientBoosting", 0.748, 0.020, 0.08),
    ])
    assert select_model(results) == "LogisticRegression"


def test_picks_clearly_better_model():
    results = _df([
        ("LogisticRegression", 0.70, 0.02, 0.01),
        ("RandomForest", 0.80, 0.02, 0.10),
        ("HistGradientBoosting", 0.74, 0.02, 0.08),
    ])
    assert select_model(results) == "RandomForest"