"""Phase 4 (1/2) : optimisation des hyperparametres par GridSearchCV.

Usage (depuis la racine du projet) :
    python -m src.tune

Le jeu de test n'est JAMAIS utilise ici : seul X_train / y_train est lu.
La regle de selection du modele final est ecrite dans select_model().
"""
from __future__ import annotations

import json
import logging
import sys
import time

import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold

from src import config
from src.preprocessing import load_data, split_data
from src.train import build_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Grilles volontairement petites (temps de calcul et lisibilite).
PARAM_GRIDS = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1, 10, 100],
    },
    "RandomForest": {
        "model__max_depth": [3, 5, 8, None],
        "model__min_samples_leaf": [1, 5, 10, 20],
        "model__max_features": ["sqrt", 0.5],
    },
    "HistGradientBoosting": {
        "model__learning_rate": [0.03, 0.1],
        "model__max_leaf_nodes": [4, 8],
        "model__min_samples_leaf": [20, 50],
        "model__l2_regularization": [0.0, 10.0],
    },
}

# Plus le rang est petit, plus le modele est simple.
COMPLEXITY_RANK = {"LogisticRegression": 1, "RandomForest": 2, "HistGradientBoosting": 3}


def select_model(results: pd.DataFrame) -> str:
    """Regle de selection (fixee avant de voir les resultats).

    1. Meilleur F1 macro en validation croisee.
    2. Candidats : modeles dont le F1 >= meilleur F1 - ecart-type du meilleur.
    3. Parmi les candidats : le plus simple, puis le plus petit ecart train-CV.
    `results` est indexe par nom de modele, avec les colonnes
    f1_cv_mean, f1_cv_std, gap_f1_train_minus_cv.
    """
    best_name = results["f1_cv_mean"].idxmax()
    threshold = results.loc[best_name, "f1_cv_mean"] - results.loc[best_name, "f1_cv_std"]
    candidates = results[results["f1_cv_mean"] >= threshold].copy()
    candidates["rank"] = [COMPLEXITY_RANK[name] for name in candidates.index]
    candidates = candidates.sort_values(["rank", "gap_f1_train_minus_cv"])
    return str(candidates.index[0])


def _plain(value):
    """Convertit les types numpy en types Python (serialisation JSON)."""
    return value.item() if hasattr(value, "item") else value


def main() -> int:
    try:
        config.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        df = load_data()
        X_train, _, y_train, _ = split_data(df)  # le test est ignore volontairement
        logger.info("Donnees d'entrainement : %s", X_train.shape)

        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        models = build_models()

        rows, best_params = [], {}
        for name, pipe in models.items():
            grid = PARAM_GRIDS[name]
            n_combos = 1
            for values in grid.values():
                n_combos *= len(values)
            logger.info("GridSearchCV : %s (%d combinaisons x %d plis)", name, n_combos, config.CV_FOLDS)

            start = time.time()
            search = GridSearchCV(
                pipe, grid, scoring="f1_macro", cv=cv, refit=True,
                return_train_score=True, n_jobs=1,
            )
            search.fit(X_train, y_train)
            elapsed = time.time() - start

            i = search.best_index_
            cvres = search.cv_results_
            rows.append({
                "model": name,
                "f1_cv_mean": cvres["mean_test_score"][i],
                "f1_cv_std": cvres["std_test_score"][i],
                "f1_train_mean": cvres["mean_train_score"][i],
                "gap_f1_train_minus_cv": cvres["mean_train_score"][i] - cvres["mean_test_score"][i],
                "search_time_s": elapsed,
                "best_params": json.dumps({k: _plain(v) for k, v in search.best_params_.items()}),
            })
            best_params[name] = {k: _plain(v) for k, v in search.best_params_.items()}
            pd.DataFrame(cvres).to_csv(
                config.REPORTS_DIR / f"grid_details_{name}.csv", index=False
            )

        results = pd.DataFrame(rows).set_index("model")
        results.to_csv(config.REPORTS_DIR / "tuning_results.csv")

        chosen = select_model(results)
        selection = {
            "selected_model": chosen,
            "best_params": best_params,
            "rule": (
                "Meilleur F1 macro CV ; candidats = F1 >= meilleur - ecart-type ; "
                "parmi eux le plus simple (LogReg < RandomForest < HistGradientBoosting), "
                "puis le plus petit ecart train-CV."
            ),
            "cv_scoring": "f1_macro",
            "random_state": config.RANDOM_STATE,
        }
        with open(config.REPORTS_DIR / "selected_model.json", "w", encoding="utf-8") as f:
            json.dump(selection, f, indent=2, ensure_ascii=False)

        with pd.option_context("display.width", 220, "display.max_columns", 20, "display.max_colwidth", 120):
            print(results.drop(columns=["best_params"]).round(4).to_string())
            print()
            for name, params in best_params.items():
                print(f"{name}: {params}")
        print(f"\nModele selectionne (regle ecrite a l'avance) : {chosen}")
        return 0
    except Exception as exc:
        logger.error("Echec : %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())