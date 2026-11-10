"""Configuration centrale du projet : chemins, colonnes, constantes."""
from pathlib import Path

# Ce fichier est dans src/ : parents[1] = racine du projet.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "student_mental_health.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

TARGET = "Mental_Health_Risk"
TARGET_ORDER = ["Low", "Medium", "High"]
ID_COL = "Student_ID"

NUMERIC_FEATURES = [
    "Age", "Year_of_Study", "Sleep_Hours", "Study_Hours_Per_Week", "GPA",
    "Social_Media_Hours", "Screen_Time_Hours", "Financial_Stress",
    "Family_Support", "Attendance_Rate",
]
# Colonnes ordinales : l'ordre des modalites est fourni explicitement.
ORDINAL_FEATURES = {
    "Physical_Activity": ["Low", "Moderate", "High"],
    "Diet_Quality": ["Poor", "Average", "Good"],
}
BINARY_FEATURES = ["Part_Time_Job", "Previous_Mental_Health_History"]
NOMINAL_FEATURES = ["Gender"]

FEATURES = (
    NUMERIC_FEATURES + list(ORDINAL_FEATURES) + BINARY_FEATURES + NOMINAL_FEATURES
)