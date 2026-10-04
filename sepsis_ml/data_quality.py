"""
Dataset Verification and Data Quality Audit Module.
Generates comprehensive diagnostics on raw PhysioNet 2019 ICU time-series data.
Adheres to Phase 2 of Golden Implementation Guide and Reyna et al. 2019 Table 1/2.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from sepsis_ml.config import (
    ALL_COLUMNS,
    DEMOGRAPHIC_VARIABLES,
    LAB_VARIABLES,
    PHYSIOLOGICAL_RANGES,
    RAW_DATA_DIR,
    TARGET_COL,
    VITAL_SIGNS,
    TABLES_DIR,
)
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.logger import get_logger

logger = get_logger("data_quality")


def audit_single_patient(filepath: Path) -> Dict:
    """Audits a single patient .psv file for schema, length, and sepsis occurrence."""
    df = pd.read_csv(filepath, sep="|")
    patient_id = filepath.stem
    hospital = "A" if "setA" in str(filepath.parent) else "B"
    
    num_rows = len(df)
    has_sepsis = int((df[TARGET_COL] == 1).any())
    num_sepsis_hours = int((df[TARGET_COL] == 1).sum())
    first_sepsis_hour = int(df.index[df[TARGET_COL] == 1][0]) if has_sepsis else -1
    
    # Check ICULOS monotonicity
    iculos_monotonic = bool(df["ICULOS"].is_monotonic_increasing)
    
    return {
        "patient_id": patient_id,
        "hospital": hospital,
        "num_rows": num_rows,
        "has_sepsis": has_sepsis,
        "num_sepsis_hours": num_sepsis_hours,
        "first_sepsis_hour": first_sepsis_hour,
        "iculos_monotonic": iculos_monotonic,
    }


def generate_dataset_audit(root_dir: Optional[Path] = None, max_patients: Optional[int] = None) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Performs full data quality audit across available raw patient files.
    
    Returns:
        Tuple of:
        - summary_metrics (Dict)
        - patient_audit_df (pd.DataFrame)
        - feature_audit_df (pd.DataFrame)
    """
    files = list(iter_patient_files(root_dir or RAW_DATA_DIR))
    if max_patients:
        files = files[:max_patients]
        
    logger.info(f"Auditing {len(files)} patient PSV files...")
    
    patient_records: List[Dict] = []
    frames: List[pd.DataFrame] = []
    
    for f in files:
        rec = audit_single_patient(f)
        patient_records.append(rec)
        frames.append(load_patient_psv(f))
        
    patient_df = pd.DataFrame(patient_records)
    combined_df = pd.concat(frames, ignore_index=True)
    
    total_patients = len(patient_df)
    septic_patients = int(patient_df["has_sepsis"].sum())
    total_hourly_rows = len(combined_df)
    septic_hourly_rows = int((combined_df[TARGET_COL] == 1).sum())
    
    # Missingness analysis
    missingness_series = combined_df[ALL_COLUMNS].isna().mean() * 100.0
    density_series = 100.0 - missingness_series
    
    vital_density = float(density_series[VITAL_SIGNS].mean())
    lab_density = float(density_series[LAB_VARIABLES].mean())
    demo_density = float(density_series[DEMOGRAPHIC_VARIABLES].mean())
    
    # Cross-hospital statistics
    hospital_stats = {}
    for h in ["A", "B"]:
        sub_pts = patient_df[patient_df["hospital"] == h]
        sub_rows = combined_df[combined_df["hospital"] == h]
        if len(sub_pts) > 0:
            hospital_stats[f"Hospital_{h}"] = {
                "patients": len(sub_pts),
                "septic_patients": int(sub_pts["has_sepsis"].sum()),
                "sepsis_patient_rate_pct": round(float(sub_pts["has_sepsis"].mean() * 100), 2),
                "total_rows": len(sub_rows),
                "mean_los_hours": round(float(sub_pts["num_rows"].mean()), 2),
                "vital_density_pct": round(float(100 - sub_rows[VITAL_SIGNS].isna().mean().mean() * 100), 2),
                "lab_density_pct": round(float(100 - sub_rows[LAB_VARIABLES].isna().mean().mean() * 100), 2),
            }
            
    # Feature audit DataFrame
    feature_rows = []
    for col in ALL_COLUMNS:
        is_vital = col in VITAL_SIGNS
        is_lab = col in LAB_VARIABLES
        is_demo = col in DEMOGRAPHIC_VARIABLES
        cat = "Vital" if is_vital else ("Lab" if is_lab else ("Demographic" if is_demo else "Target"))
        
        missing_pct = float(missingness_series[col])
        non_null_count = int(combined_df[col].notna().sum())
        
        # Outlier counts using physiological boundaries
        out_of_bounds = 0
        if col in PHYSIOLOGICAL_RANGES:
            low, high = PHYSIOLOGICAL_RANGES[col]
            valid_vals = combined_df[col].dropna()
            out_of_bounds = int(((valid_vals < low) | (valid_vals > high)).sum())
            
        feature_rows.append({
            "feature": col,
            "category": cat,
            "missing_pct": round(missing_pct, 2),
            "density_pct": round(100.0 - missing_pct, 2),
            "non_null_count": non_null_count,
            "out_of_bounds_count": out_of_bounds,
            "min": round(float(combined_df[col].min()), 2) if non_null_count > 0 else np.nan,
            "mean": round(float(combined_df[col].mean()), 2) if non_null_count > 0 else np.nan,
            "max": round(float(combined_df[col].max()), 2) if non_null_count > 0 else np.nan,
        })
        
    feature_df = pd.DataFrame(feature_rows)
    
    summary_metrics = {
        "total_patients_audited": total_patients,
        "total_septic_patients": septic_patients,
        "patient_sepsis_prevalence_pct": round((septic_patients / total_patients) * 100, 2) if total_patients else 0,
        "total_hourly_records": total_hourly_rows,
        "total_sepsis_positive_hours": septic_hourly_rows,
        "hourly_sepsis_prevalence_pct": round((septic_hourly_rows / total_hourly_rows) * 100, 2) if total_hourly_rows else 0,
        "mean_stay_length_hours": round(float(patient_df["num_rows"].mean()), 2),
        "median_stay_length_hours": round(float(patient_df["num_rows"].median()), 2),
        "min_stay_length_hours": int(patient_df["num_rows"].min()),
        "max_stay_length_hours": int(patient_df["num_rows"].max()),
        "vital_signs_density_pct": round(vital_density, 2),
        "lab_variables_density_pct": round(lab_density, 2),
        "demographics_density_pct": round(demo_density, 2),
        "hospital_comparison": hospital_stats,
    }
    
    # Save audit artifacts
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    report_path = TABLES_DIR / "dataset_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=2)
        
    feature_df.to_csv(TABLES_DIR / "feature_audit.csv", index=False)
    patient_df.to_csv(TABLES_DIR / "patient_audit.csv", index=False)
    
    logger.info(f"Audit completed. Summary written to {report_path}")
    return summary_metrics, patient_df, feature_df
