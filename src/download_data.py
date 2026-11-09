"""Telecharge le dataset Kaggle et le copie dans data/raw/.

Usage (depuis la racine du projet) :
    python src/download_data.py          # saute si le CSV existe deja
    python src/download_data.py --force  # retelecharge et ecrase
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import kagglehub

DATASET_HANDLE = "maqdadali/mental-health-risk-prediction-dataset-students"
EXPECTED_FILE = "student_mental_health.csv"

# Ce fichier est dans src/ : parents[0] = src, parents[1] = racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_dataset(force: bool = False) -> Path:
    """Telecharge le dataset Kaggle et copie ses fichiers dans data/raw."""
    if not (PROJECT_ROOT / "requirements.txt").exists():
        raise RuntimeError(
            f"Racine du projet incorrecte : {PROJECT_ROOT} (requirements.txt introuvable)."
        )

    target = RAW_DATA_DIR / EXPECTED_FILE
    if target.exists() and not force:
        logger.info("%s existe deja, telechargement ignore (utiliser --force).", target)
        return RAW_DATA_DIR

    logger.info("Telechargement de %s ...", DATASET_HANDLE)
    downloaded_path = Path(kagglehub.dataset_download(DATASET_HANDLE))

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(downloaded_path, RAW_DATA_DIR, dirs_exist_ok=True)

    if not target.exists():
        raise FileNotFoundError(f"Fichier attendu absent apres copie : {target}")
    logger.info("Dataset disponible dans %s", RAW_DATA_DIR)
    return RAW_DATA_DIR


def main() -> int:
    parser = argparse.ArgumentParser(description="Telecharge le dataset Kaggle.")
    parser.add_argument("--force", action="store_true", help="ecrase les donnees existantes")
    args = parser.parse_args()
    try:
        download_dataset(force=args.force)
    except Exception as exc:  # erreur reseau, Kaggle, disque, chemin...
        logger.error("Echec du telechargement : %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())