import json

from src import config
from src.evaluate import build_final_pipeline, error_breakdown
from src.preprocessing import load_data, split_data


def test_error_breakdown_counts():
    y_true = ["Low", "Low", "Medium", "High", "High"]
    y_pred = ["Low", "Medium", "Medium", "Low", "High"]
    assert error_breakdown(y_true, y_pred) == {
        "correct": 3, "adjacent_errors": 1, "extreme_errors_low_high": 1
    }


def test_build_final_pipeline_applies_selected_params():
    selection = {
        "selected_model": "LogisticRegression",
        "best_params": {"LogisticRegression": {"model__C": 100}},
    }
    name, pipe = build_final_pipeline(selection)
    assert name == "LogisticRegression"
    assert pipe.get_params()["model__C"] == 100


def test_selected_pipeline_fits_on_train_rows_only():
    # N'utilise QUE le jeu d'entrainement (le test reste intact).
    X_train, _, y_train, _ = split_data(load_data())
    with open(config.REPORTS_DIR / "selected_model.json", encoding="utf-8") as f:
        selection = json.load(f)
    _, pipe = build_final_pipeline(selection)
    pipe.fit(X_train.iloc[:300], y_train.iloc[:300])
    pred = pipe.predict(X_train.iloc[300:320])
    assert set(pred) <= set(config.TARGET_ORDER)