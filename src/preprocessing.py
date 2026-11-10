"""Chargement, separation train/test et pipeline de pretraitement.

Regle anti-fuite : la separation est faite AVANT tout ajustement ; le
ColumnTransformer est ensuite ajuste uniquement sur le jeu d'entrainement
(dans un Pipeline, donc refait a l'interieur de chaque pli de validation croisee).
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from src import config

logger = logging.getLogger(__name__)


def load_data(path: Path = config.RAW_DATA_PATH) -> pd.DataFrame:
    """Charge le CSV brut et verifie sa structure minimale."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lancer : python src/download_data.py"
        )
    df = pd.read_csv(path)

    expected = set(config.FEATURES) | {config.TARGET, config.ID_COL}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes absentes du dataset : {sorted(missing)}")
    if df[config.TARGET].isna().any():
        raise ValueError("La cible contient des valeurs manquantes.")
    unknown = set(df[config.TARGET].unique()) - set(config.TARGET_ORDER)
    if unknown:
        raise ValueError(f"Modalites inattendues dans la cible : {sorted(unknown)}")
    return df


def split_data(df: pd.DataFrame):
    """Separation 80/20 stratifiee. Retourne X_train, X_test, y_train, y_test.

    Student_ID et la cible ne font pas partie des variables explicatives.
    """
    X = df[config.FEATURES]
    y = df[config.TARGET]
    return train_test_split(
        X,
        y,
        test_size=config.TEST_SIZE,
        stratify=y,
        random_state=config.RANDOM_STATE,
    )


def build_preprocessor(scale_numeric: bool) -> ColumnTransformer:
    """Construit le pretraitement (non ajuste).

    scale_numeric=True pour la regression logistique ; False pour les arbres.
    """
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))

    ordinal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=list(config.ORDINAL_FEATURES.values()))),
    ])
    binary = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(categories=[["No", "Yes"]] * len(config.BINARY_FEATURES))),
    ])
    nominal = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), config.NUMERIC_FEATURES),
            ("ord", ordinal, list(config.ORDINAL_FEATURES)),
            ("bin", binary, config.BINARY_FEATURES),
            ("nom", nominal, config.NOMINAL_FEATURES),
        ],
        remainder="drop",
    )