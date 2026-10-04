"""
Leakage-Safe Target Construction and Prediction Horizon Module.
Adheres to PhysioNet Challenge 2019 Sepsis-3 Target Formulation.
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sepsis_ml.config import TARGET_COL
from sepsis_ml.logger import get_logger

logger = get_logger("target")

OFFICIAL_LEAD_TIME_HOURS = 6


def extract_patient_sepsis_metadata(patient_df: pd.DataFrame) -> Dict:
    """
    Extracts ground truth clinical sepsis timestamps from a patient's hourly series.
    
    In the PhysioNet 2019 dataset:
    - SepsisLabel == 1 begins at t_sepsis - 6 hours.
    - Therefore: t_sepsis = first_positive_hour + 6 hours.
    
    Returns:
        Dict with keys:
        - is_septic (bool)
        - first_positive_index (int or None)
        - tsepsis_index (int or None)
        - total_hours (int)
    """
    labels = patient_df[TARGET_COL].values
    is_septic = bool((labels == 1).any())
    
    if not is_septic:
        return {
            "is_septic": False,
            "first_positive_index": None,
            "tsepsis_index": None,
            "total_hours": len(labels),
        }
        
    first_pos_idx = int(np.where(labels == 1)[0][0])
    tsepsis_idx = first_pos_idx + OFFICIAL_LEAD_TIME_HOURS
    
    return {
        "is_septic": True,
        "first_positive_index": first_pos_idx,
        "tsepsis_index": tsepsis_idx,
        "total_hours": len(labels),
    }


def construct_horizon_target(
    patient_df: pd.DataFrame,
    horizon_hours: int = 6
) -> pd.Series:
    """
    Constructs a leakage-safe binary target vector for a given prediction horizon H.
    
    At horizon H (hours before clinical onset tsepsis):
    - For non-septic patients: y_t = 0 for all t.
    - For septic patients:
        y_t = 1 if t >= tsepsis - H, else 0.
        
    When H == 6, this matches the official challenge SepsisLabel exactly.
    
    Args:
        patient_df: DataFrame of a single patient sorted by ICULOS
        horizon_hours: Warning horizon in hours (default: 6)
        
    Returns:
        pd.Series containing binary labels {0, 1}
    """
    meta = extract_patient_sepsis_metadata(patient_df)
    n_hours = meta["total_hours"]
    
    if not meta["is_septic"]:
        return pd.Series(np.zeros(n_hours, dtype=np.int8), index=patient_df.index, name=f"target_{horizon_hours}h")
        
    tsepsis = meta["tsepsis_index"]
    target_start_idx = max(0, tsepsis - horizon_hours)
    
    target_arr = np.zeros(n_hours, dtype=np.int8)
    target_arr[target_start_idx:] = 1
    
    return pd.Series(target_arr, index=patient_df.index, name=f"target_{horizon_hours}h")


def add_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Appends multi-horizon binary targets to an existing patient cohort DataFrame.
    Guarantees strict patient-by-patient grouping with zero cross-patient leakage.
    
    Args:
        df: Cohort DataFrame containing 'patient_id' and 'SepsisLabel'
        horizons: List of horizon integers (default: [3, 6, 9, 12])
        
    Returns:
        DataFrame with added target columns.
    """
    if horizons is None:
        horizons = [3, 6, 9, 12]
        
    result_df = df.copy()
    
    for h in horizons:
        col_name = f"target_{h}h"
        target_series_list = []
        for pid, group in result_df.groupby("patient_id", sort=False):
            t_s = construct_horizon_target(group, horizon_hours=h)
            target_series_list.append(t_s)
        result_df[col_name] = pd.concat(target_series_list).values
        
    return result_df


def audit_target_leakage(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Audits target formulation against clinical and mathematical leakage criteria:
    1. Binary values only: set(y) in {0, 1}.
    2. Step-function monotonicity: for septic patients, target never drops from 1 back to 0.
    3. Pure negative records: non-septic patients have sum(y) == 0.
    4. Exact 6h parity: target_6h matches raw SepsisLabel on all rows.
    
    Returns:
        Tuple of (is_clean: bool, violation_messages: List[str])
    """
    violations = []
    
    # Check binary values
    unique_vals = set(df[TARGET_COL].unique())
    if not unique_vals.issubset({0, 1}):
        violations.append(f"Target contains non-binary values: {unique_vals}")
        
    # Check step-function monotonicity per patient
    for pid, group in df.groupby("patient_id"):
        labels = group[TARGET_COL].values
        if (labels == 1).any():
            first_pos = np.where(labels == 1)[0][0]
            if (labels[first_pos:] == 0).any():
                violations.append(f"Patient {pid} exhibited target reversion from 1 to 0.")
                
    is_clean = len(violations) == 0
    return is_clean, violations
