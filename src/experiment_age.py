"""Experience : la variable Age est-elle utile ? (CV 5 plis, train seulement)

Usage : python -m src.experiment_age
Regle fixee a l'avance : on retire Age si, pour AUCUN des 3 modeles, le F1 macro
ne baisse de plus de 0.01 apres retrait. Le jeu de test n'est jamais utilise.
"""
from __future__ import annotations

import logging
import sys

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

from src import config
from src.preprocessing import load_data, split_data
from src.train import build_models

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

THRESHOLD = 0.01


def run(drop_age: bool, X_train, y_train, cv) -> dict:
    """F1 macro CV de chaque modele, avec ou sans Age."""
    original = config.NUMERIC_FEATURES
    if drop_age:
        config.NUMERIC_FEATURES = [c for c in original if c != "Age"]
    try:
        scores = {}
        for name, pipe in build_models().items():
            res = cross_validate(pipe, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=1)
            scores[name] = (res["test_score"].mean(), res["test_score"].std())
        return scores
    finally:
        config.NUMERIC_FEATURES = original


def main() -> int:
    try:
        X_train, _, y_train, _ = split_data(load_data())  # le test est ignore
        cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
        logger.info("Avec Age ...")
        with_age = run(False, X_train, y_train, cv)
        logger.info("Sans Age ...")
        without_age = run(True, X_train, y_train, cv)

        rows = []
        for name in with_age:
            rows.append({
                "model": name,
                "f1_with_age": with_age[name][0], "std_with_age": with_age[name][1],
                "f1_without_age": without_age[name][0], "std_without_age": without_age[name][1],
                "delta_without_minus_with": without_age[name][0] - with_age[name][0],
            })
        out = pd.DataFrame(rows).set_index("model")
        out.to_csv(config.REPORTS_DIR / "age_experiment.csv")
        print(out.round(4).to_string())
        drop = bool((out["delta_without_minus_with"] >= -THRESHOLD).all())
        print(f"\nRegle (retirer Age si aucune baisse > {THRESHOLD}) -> retirer Age : {drop}")
        return 0
    except Exception as exc:
        logger.error("Echec : %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())