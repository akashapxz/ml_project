"""
Central Configuration and Schema Definitions for Sepsis ML Project.
Source of Truth: PhysioNet/Computing in Cardiology Challenge 2019 & Golden Implementation Guide.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


# Base Workspace Directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"
MODELS_DIR = OUTPUTS_DIR / "models"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
REPORTS_DIR = PROJECT_ROOT / "reports"

# 40 Clinical Variables strictly defined in Reyna et al. (Table 1)
VITAL_SIGNS: List[str] = [
    "HR",           # Heart rate (bpm)
    "O2Sat",        # Pulse oximetry (%)
    "Temp",         # Temperature (deg C)
    "SBP",          # Systolic BP (mmHg)
    "MAP",          # Mean arterial pressure (mmHg)
    "DBP",          # Diastolic BP (mmHg)
    "Resp",         # Respiration rate (breaths/min)
    "EtCO2",        # End-tidal CO2 (mmHg)
]

LAB_VARIABLES: List[str] = [
    "BaseExcess",        # Excess bicarbonate (mmol/L)
    "HCO3",              # Bicarbonate (mmol/L)
    "FiO2",              # Fraction of inspired oxygen (%)
    "pH",                # Arterial pH
    "PaCO2",             # Partial pressure of CO2 (mmHg)
    "SaO2",              # Arterial oxygen saturation (%)
    "AST",               # Aspartate transaminase (IU/L)
    "BUN",               # Blood urea nitrogen (mg/dL)
    "Alkalinephos",      # Alkaline phosphatase (IU/L)
    "Calcium",           # Serum Calcium (mg/dL)
    "Chloride",          # Serum Chloride (mmol/L)
    "Creatinine",        # Serum Creatinine (mg/dL)
    "Bilirubin_direct",  # Direct bilirubin (mg/dL)
    "Glucose",           # Serum glucose (mg/dL)
    "Lactate",           # Lactic acid (mg/dL)
    "Magnesium",         # Serum Magnesium (mg/dL)
    "Phosphate",         # Serum Phosphate (mg/dL)
    "Potassium",         # Serum Potassium (mmol/L)
    "Bilirubin_total",   # Total bilirubin (mg/dL)
    "TroponinI",         # Cardiac Troponin I (ng/mL)
    "Hct",               # Hematocrit (%)
    "Hgb",               # Hemoglobin (g/dL)
    "PTT",               # Partial thromboplastin time (sec)
    "WBC",               # Leukocyte count (count/L)
    "Fibrinogen",        # Fibrinogen concentration (mg/dL)
    "Platelets",         # Platelet count (count/mL)
]

DEMOGRAPHIC_VARIABLES: List[str] = [
    "Age",               # Age (years)
    "Gender",            # Female (0) or Male (1)
    "Unit1",             # ICU Unit 1 (MICU) indicator
    "Unit2",             # ICU Unit 2 (SICU) indicator
    "HospAdmTime",       # Time between hospital and ICU admission (hours)
    "ICULOS",            # ICU length of stay (hours since ICU admission)
]

TARGET_COL: str = "SepsisLabel"

ALL_CLINICAL_VARIABLES: List[str] = VITAL_SIGNS + LAB_VARIABLES + DEMOGRAPHIC_VARIABLES
ALL_COLUMNS: List[str] = ALL_CLINICAL_VARIABLES + [TARGET_COL]

# Standard Physiological Normal Ranges for Clinical Sanity Auditing
PHYSIOLOGICAL_RANGES: Dict[str, tuple] = {
    "HR": (20.0, 250.0),
    "O2Sat": (40.0, 100.0),
    "Temp": (28.0, 44.0),
    "SBP": (30.0, 300.0),
    "MAP": (20.0, 220.0),
    "DBP": (15.0, 200.0),
    "Resp": (2.0, 80.0),
    "Glucose": (10.0, 1000.0),
    "pH": (6.5, 7.8),
    "Lactate": (0.1, 30.0),
    "Potassium": (1.0, 12.0),
    "Creatinine": (0.1, 25.0),
}


@dataclass
class ProjectConfig:
    """Immutable experiment and reproducibility configuration container."""
    seed: int = 42
    test_size: float = 0.15
    val_size: float = 0.15
    train_size: float = 0.70
    
    # Feature Ablations
    feature_families: List[str] = field(default_factory=lambda: ["F1", "F2", "F3", "F4"])
    
    # Rolling window sizes (hours)
    rolling_windows: List[int] = field(default_factory=lambda: [3, 6, 12])
    
    # Prediction horizons for sensitivity analysis (hours)
    prediction_horizons: List[int] = field(default_factory=lambda: [3, 6, 9, 12])
    
    # Challenge Utility Parameters (Reyna et al. 2019)
    utility_dt_early: float = -12.0  # hours relative to tsepsis
    utility_dt_optimal: float = -6.0
    utility_dt_late: float = 3.0
    utility_max_reward: float = 1.0
    utility_max_penalty: float = -2.0
    utility_fp_penalty: float = -0.05
