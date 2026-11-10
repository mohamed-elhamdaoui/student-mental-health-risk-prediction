"""Phase 3 : comparaison de trois modeles par validation croisee (train seulement).

Usage (depuis la racine du projet) :
    python -m src.train

Le jeu de test est mis de cote et JAMAIS utilise ici.
"""
from __future__ import annotations

import logging
import sys

import matplotlib

matplotlib.use("Agg")  # pas de fenetre : sauvegarde directe des figures
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline

from src import config
from src.preprocessing import build_preprocessor, load_data, split_data

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

SCORING = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    "roc_auc_ovr": "roc_auc_ovr",
}


def build_models() -> dict[str, Pipeline]:
    """Trois pipelines complets (pretraitement + modele)."""
    rs = config.RANDOM_STATE
    return {
        "LogisticRegression": Pipeline([
            ("prep", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=2000, random_state=rs)),
        ]),
        "RandomForest": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1)),
        ]),
        "HistGradientBoosting": Pipeline([
            ("prep", build_preprocessor(scale_numeric=False)),
            ("model", HistGradientBoostingClassifier(random_state=rs)),
        ]),
    }


def evaluate_models(models, X_train, y_train, cv) -> pd.DataFrame:
    """Validation croisee stratifiee ; moyenne et ecart-type de chaque metrique."""
    rows = []
    for name, pipe in models.items():
        logger.info("Validation croisee : %s", name)
        res = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=SCORING,
            return_train_score=True, n_jobs=1,
        )
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = res[f"test_{metric}"].mean()
            row[f"{metric}_std"] = res[f"test_{metric}"].std()
        row["train_f1_macro_mean"] = res["train_f1_macro"].mean()
        row["gap_f1_train_minus_cv"] = row["train_f1_macro_mean"] - row["f1_macro_mean"]
        row["fit_time_s"] = res["fit_time"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def plot_comparison(results: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(results.index, results["f1_macro_mean"], yerr=results["f1_macro_std"], capsize=5)
    ax.set_ylabel("F1 macro (validation croisee)")
    ax.set_title("Comparaison des modeles (train, 5 plis)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "06_cv_model_comparison.png", dpi=120)
    plt.close(fig)


def plot_confusions(models, X_train, y_train, cv) -> None:
    """Matrices de confusion issues de predictions hors-pli (donnees train)."""
    fig, axes = plt.subplots(1, len(models), figsize=(5 * len(models), 4.5))
    for ax, (name, pipe) in zip(axes, models.items()):
        pred = cross_val_predict(pipe, X_train, y_train, cv=cv)
        ConfusionMatrixDisplay.from_predictions(
            y_train, pred, labels=config.TARGET_ORDER, ax=ax, colorbar=False
        )
        ax.set_title(name)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "07_cv_confusion_matrices.png", dpi=120)
    plt.close(fig)


def main() -> int:
    try:
        config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        df = load_data()
        X_train, X_test, y_train, y_test = split_data(df)
        logger.info("Train : %s | Test (mis de cote) : %s", X_train.shape, X_test.shape)

        # Sauvegarde de la separation (data/processed est ignore par Git).
        X_train.assign(**{config.TARGET: y_train}).to_csv(config.PROCESSED_DIR / "train.csv")
        X_test.assign(**{config.TARGET: y_test}).to_csv(config.PROCESSED_DIR / "test.csv")

        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        models = build_models()
        results = evaluate_models(models, X_train, y_train, cv)
        results.to_csv(config.REPORTS_DIR / "cv_results.csv")

        plot_comparison(results)
        plot_confusions(models, X_train, y_train, cv)

        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(results.round(4).T.to_string())
        return 0
    except Exception as exc:
        logger.error("Echec : %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())