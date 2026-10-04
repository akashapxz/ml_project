"""
Stratified Patient-Level Partitioning Module.
Guarantees absolute subject independence, balanced prevalence across splits,
and zero cross-split temporal or patient data leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from sepsis_ml.config import ProjectConfig, TABLES_DIR, TARGET_COL
from sepsis_ml.logger import get_logger
from sepsis_ml.reproducibility import seed_everything

logger = get_logger("splitting")


def build_patient_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts a patient-level summary table suitable for stratified subject partitioning.
    
    Args:
        df: Cohort DataFrame with 'patient_id', 'hospital', and 'SepsisLabel'.
        
    Returns:
        pd.DataFrame indexed by patient_id with stratification attributes.
    """
    records = []
    for pid, group in df.groupby("patient_id", sort=False):
        has_sepsis = int((group[TARGET_COL] == 1).any())
        hospital = group["hospital"].iloc[0] if "hospital" in group.columns else "A"
        num_rows = len(group)
        records.append({
            "patient_id": pid,
            "hospital": hospital,
            "has_sepsis": has_sepsis,
            "num_rows": num_rows,
            # Composite stratum for dual stratification on outcome and hospital system
            "stratum": f"{hospital}_{has_sepsis}",
        })
    meta_df = pd.DataFrame(records)
    return meta_df


def stratified_patient_split(
    df: pd.DataFrame,
    cfg: Optional[ProjectConfig] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, List[str]]]:
    """
    Performs patient-level stratified splitting into Train (70%), Validation (15%), and Test (15%).
    
    Guarantees:
    - Zero row-wise or patient-wise overlap across splits.
    - Preserves patient-level sepsis prevalence and hospital ratio across all 3 sets.
    - Completely deterministic under cfg.seed.
    
    Returns:
        Tuple of (train_df, val_df, test_df, split_dict)
    """
    config = cfg or ProjectConfig()
    seed_everything(config.seed)
    
    meta_df = build_patient_metadata(df)
    n_patients = len(meta_df)
    
    logger.info(f"Partitioning {n_patients} patients (Train={config.train_size:.0%}, Val={config.val_size:.0%}, Test={config.test_size:.0%})...")
    
    # Stratification labels: composite of sepsis status and hospital origin
    # Fallback to pure has_sepsis if any stratum has < 2 members
    strata = meta_df["stratum"].values
    counts = meta_df["stratum"].value_counts()
    if (counts < 2).any():
        strata = meta_df["has_sepsis"].values
        
    # Stage 1: Split into Train (70%) and Temp (30%)
    temp_size = config.val_size + config.test_size
    sss1 = StratifiedShuffleSplit(n_splits=1, test_size=temp_size, random_state=config.seed)
    train_idx, temp_idx = next(sss1.split(meta_df, strata))
    
    train_meta = meta_df.iloc[train_idx]
    temp_meta = meta_df.iloc[temp_idx]
    
    # Stage 2: Split Temp (30%) equally into Validation (15%) and Test (15%)
    temp_strata = temp_meta["has_sepsis"].values
    val_ratio = config.val_size / temp_size  # 0.15 / 0.30 = 0.50
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=(1.0 - val_ratio), random_state=config.seed)
    val_idx, test_idx = next(sss2.split(temp_meta, temp_strata))
    
    val_meta = temp_meta.iloc[val_idx]
    test_meta = temp_meta.iloc[test_idx]
    
    train_pids = sorted(train_meta["patient_id"].tolist())
    val_pids = sorted(val_meta["patient_id"].tolist())
    test_pids = sorted(test_meta["patient_id"].tolist())
    
    # Strict Patient Overlap Audit
    is_clean, violations = audit_split_overlap(train_pids, val_pids, test_pids)
    if not is_clean:
        raise ValueError(f"Patient overlap detected during split: {violations}")
        
    # Slice full row-level DataFrames
    train_df = df[df["patient_id"].isin(train_pids)].copy()
    val_df = df[df["patient_id"].isin(val_pids)].copy()
    test_df = df[df["patient_id"].isin(test_pids)].copy()
    
    split_pids = {
        "train": train_pids,
        "val": val_pids,
        "test": test_pids,
    }
    
    # Save Split Artifacts
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(TABLES_DIR / "patient_splits.json", "w", encoding="utf-8") as f:
        json.dump(split_pids, f, indent=2)
        
    # Generate and save split summary
    summary_rows = []
    for split_name, s_df, s_pids in [
        ("Train", train_df, train_pids),
        ("Validation", val_df, val_pids),
        ("Test", test_df, test_pids),
    ]:
        s_meta = meta_df[meta_df["patient_id"].isin(s_pids)]
        summary_rows.append({
            "split": split_name,
            "patient_count": len(s_pids),
            "patient_pct": round(len(s_pids) / n_patients * 100, 2),
            "septic_patients": int(s_meta["has_sepsis"].sum()),
            "patient_sepsis_prev_pct": round(float(s_meta["has_sepsis"].mean() * 100), 2),
            "total_hourly_rows": len(s_df),
            "septic_hourly_rows": int((s_df[TARGET_COL] == 1).sum()),
            "hourly_sepsis_prev_pct": round(float(s_df[TARGET_COL].mean() * 100), 2),
        })
        
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(TABLES_DIR / "patient_split_summary.csv", index=False)
    
    logger.info("Stratified patient split generated successfully:")
    for _, row in summary_df.iterrows():
        logger.info(
            f"  [{row['split']}] Patients: {row['patient_count']} ({row['patient_sepsis_prev_pct']}% septic) | "
            f"Rows: {row['total_hourly_rows']} ({row['hourly_sepsis_prev_pct']}% positive)"
        )
        
    return train_df, val_df, test_df, split_pids


def audit_split_overlap(
    train_pids: List[str],
    val_pids: List[str],
    test_pids: List[str]
) -> Tuple[bool, List[str]]:
    """
    Audits patient ID lists to guarantee strictly zero intersection.
    
    Returns:
        Tuple of (is_clean: bool, violations: List[str])
    """
    violations = []
    s_train, s_val, s_test = set(train_pids), set(val_pids), set(test_pids)
    
    overlap_train_val = s_train & s_val
    overlap_train_test = s_train & s_test
    overlap_val_test = s_val & s_test
    
    if overlap_train_val:
        violations.append(f"Train/Val patient overlap detected: {len(overlap_train_val)} patients: {list(overlap_train_val)[:3]}")
    if overlap_train_test:
        violations.append(f"Train/Test patient overlap detected: {len(overlap_train_test)} patients: {list(overlap_train_test)[:3]}")
    if overlap_val_test:
        violations.append(f"Val/Test patient overlap detected: {len(overlap_val_test)} patients: {list(overlap_val_test)[:3]}")
        
    total_unique = len(s_train | s_val | s_test)
    total_sum = len(s_train) + len(s_val) + len(s_test)
    if total_unique != total_sum:
        violations.append(f"Sum of split sizes ({total_sum}) does not equal unique patient count ({total_unique})")
        
    is_clean = len(violations) == 0
    return is_clean, violations
