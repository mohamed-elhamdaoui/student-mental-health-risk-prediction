"""Phase 4 (2/2) : entrainement final, UNE evaluation sur le test, sauvegarde.

Usage (depuis la racine du projet) :
    python -m src.evaluate

Le jeu de test est lu ici et seulement ici. Garde-fou : si
reports/test_results.json existe, le script refuse de relancer l'evaluation
(--force n'est a utiliser qu'en connaissance de cause : le test serait reutilise).
"""
from __future__ import annotations

import argparse
import json
import logging
import sys

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src import config
from src.preprocessing import load_data, split_data
from src.train import build_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SELECTION_PATH = config.REPORTS_DIR / "selected_model.json"
TUNING_PATH = config.REPORTS_DIR / "tuning_results.csv"
RESULTS_PATH = config.REPORTS_DIR / "test_results.json"
PREDICTIONS_PATH = config.REPORTS_DIR / "test_predictions.csv"
COEF_PATH = config.REPORTS_DIR / "logreg_coefficients.csv"
MODEL_PATH = config.MODELS_DIR / "final_pipeline.joblib"
META_PATH = config.MODELS_DIR / "final_pipeline_metadata.json"


def build_final_pipeline(selection: dict):
    """Reconstruit le pipeline choisi avec ses meilleurs hyperparametres (non ajuste)."""
    name = selection["selected_model"]
    pipe = build_models()[name]
    pipe.set_params(**selection["best_params"][name])
    return name, pipe


def error_breakdown(y_true, y_pred) -> dict:
    """Compte les erreurs entre classes adjacentes vs entre Low et High."""
    rank = {c: i for i, c in enumerate(config.TARGET_ORDER)}
    t = np.array([rank[v] for v in np.asarray(y_true)])
    p = np.array([rank[v] for v in np.asarray(y_pred)])
    dist = np.abs(t - p)
    return {
        "correct": int((dist == 0).sum()),
        "adjacent_errors": int((dist == 1).sum()),
        "extreme_errors_low_high": int((dist == 2).sum()),
    }


def bootstrap_f1_ci(y_true, y_pred, n_boot: int = 1000, seed: int = config.RANDOM_STATE):
    """Intervalle de confiance a 95 % (bootstrap) du F1 macro, sans reentrainement."""
    rng = np.random.default_rng(seed)
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(f1_score(y_true[idx], y_pred[idx], average="macro"))
    return float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5))


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluation finale unique sur le jeu de test.")
    parser.add_argument("--force", action="store_true", help="reutiliser le test (deconseille)")
    args = parser.parse_args()
    try:
        if RESULTS_PATH.exists() and not args.force:
            raise RuntimeError(
                f"{RESULTS_PATH} existe deja : le test a deja ete evalue. "
                "Ne pas le reutiliser pour modifier le modele (--force a vos risques)."
            )
        with open(SELECTION_PATH, encoding="utf-8") as f:
            selection = json.load(f)
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        X_train, X_test, y_train, y_test = split_data(load_data())
        name, pipe = build_final_pipeline(selection)
        logger.info("Modele final : %s | entrainement sur %d lignes", name, len(X_train))

        # 1. Entrainement final sur TOUT le train, puis sauvegarde du pipeline complet.
        pipe.fit(X_train, y_train)
        joblib.dump(pipe, MODEL_PATH)
        meta = {
            "model": name,
            "params": selection["best_params"][name],
            "features": config.FEATURES,
            "target_classes": [str(c) for c in pipe.classes_],
            "trained_on": "train split only (80 %)",
            "random_state": config.RANDOM_STATE,
            "sklearn_version": sklearn.__version__,
            "disclaimer": "Prototype educatif sur donnees synthetiques ; ni diagnostic ni depistage.",
        }
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        # 2. Evaluation UNIQUE sur le test, avec le pipeline recharge depuis le disque.
        model = joblib.load(MODEL_PATH)
        y_pred = model.predict(X_test)
        proba = model.predict_proba(X_test)
        classes = [str(c) for c in model.classes_]
        y_true_arr = y_test.to_numpy()

        report = classification_report(
            y_test, y_pred, labels=config.TARGET_ORDER, output_dict=True, zero_division=0
        )
        per_class = {
            c: {k: float(report[c][k]) for k in ("precision", "recall", "f1-score", "support")}
            for c in config.TARGET_ORDER
        }
        dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
        dummy_pred = dummy.predict(X_test)
        confidence = proba.max(axis=1)
        correct = y_pred == y_true_arr
        lo, hi = bootstrap_f1_ci(y_test, y_pred)

        cv_ref = None
        if TUNING_PATH.exists():
            cv_ref = float(pd.read_csv(TUNING_PATH, index_col="model").loc[name, "f1_cv_mean"])

        results = {
            "model": name,
            "best_params": selection["best_params"][name],
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision_macro": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
            "recall_macro": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
            "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
            "f1_macro_ci95_bootstrap": [lo, hi],
            "roc_auc_ovr": float(roc_auc_score(y_test, proba, multi_class="ovr", labels=classes)),
            "f1_macro_cv_reference": cv_ref,
            "per_class": per_class,
            "confusion_matrix": {
                "labels": config.TARGET_ORDER,
                "matrix": confusion_matrix(y_test, y_pred, labels=config.TARGET_ORDER).tolist(),
            },
            "error_breakdown": error_breakdown(y_test, y_pred),
            "baseline_majority_class": {
                "accuracy": float(accuracy_score(y_test, dummy_pred)),
                "f1_macro": float(f1_score(y_test, dummy_pred, average="macro", zero_division=0)),
            },
            "mean_confidence_correct": float(confidence[correct].mean()),
            "mean_confidence_wrong": float(confidence[~correct].mean()),
        }
        with open(RESULTS_PATH, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        # 3. Sorties complementaires (predictions, figure, coefficients).
        preds = pd.DataFrame(
            {"y_true": y_true_arr, "y_pred": y_pred,
             **{f"proba_{c}": proba[:, i] for i, c in enumerate(classes)}},
            index=y_test.index,
        )
        preds.index.name = "row_index"
        preds.to_csv(PREDICTIONS_PATH)

        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_test, y_pred, labels=config.TARGET_ORDER, ax=ax, colorbar=False
        )
        ax.set_title(f"Matrice de confusion - jeu de test ({name})")
        fig.tight_layout()
        fig.savefig(config.FIGURES_DIR / "08_test_confusion_matrix.png", dpi=120)
        plt.close(fig)

        estimator = model.named_steps["model"]
        if hasattr(estimator, "coef_"):
            names = model.named_steps["prep"].get_feature_names_out()
            pd.DataFrame(estimator.coef_.T, index=names, columns=classes).to_csv(COEF_PATH)

        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:
        logger.error("Echec : %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())