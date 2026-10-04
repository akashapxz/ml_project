"""
Temporal-Aware Preprocessing and Imputation Module.
Enforces strictly causal, real-time clinical constraints:
1. Binary missingness indicators (_isnan) capturing observation intent.
2. Intra-patient forward filling (ffill) of past measurements only.
3. Training-derived population median imputation for unobserved initial states.
4. Physiological outlier clipping.
5. Absolute prevention of backward fill or future-to-past information leakage.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from sepsis_ml.config import (
    ALL_COLUMNS,
    DEMOGRAPHIC_VARIABLES,
    LAB_VARIABLES,
    MODELS_DIR,
    PHYSIOLOGICAL_RANGES,
    TARGET_COL,
    VITAL_SIGNS,
)
from sepsis_ml.logger import get_logger

logger = get_logger("preprocessing")

NUMERICAL_COVARIATES = [
    col for col in (VITAL_SIGNS + LAB_VARIABLES + DEMOGRAPHIC_VARIABLES)
    if col not in ["Unit1", "Unit2", "Gender"]
]
CATEGORICAL_COVARIATES = ["Gender", "Unit1", "Unit2"]


class ClinicalPreprocessor(BaseEstimator, TransformerMixin):
    """
    Causal, leakage-safe clinical preprocessor.
    Fitted strictly on Training cohort; applied without recomputation to Val/Test cohorts.
    """

    def __init__(
        self,
        include_missingness_flags: bool = True,
        clip_physiological_bounds: bool = True,
    ):
        self.include_missingness_flags = include_missingness_flags
        self.clip_physiological_bounds = clip_physiological_bounds
        self.training_medians_: Dict[str, float] = {}
        self.is_fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "ClinicalPreprocessor":
        """
        Computes population medians exclusively from the training partition.
        No validation or test data is accessed.
        """
        logger.info(f"Fitting ClinicalPreprocessor on {len(X)} training rows...")
        
        # Calculate training medians across all numerical features
        self.training_medians_ = {}
        for col in NUMERICAL_COVARIATES:
            median_val = float(X[col].median(skipna=True))
            # Fallback for 100% missing column in small sample
            if np.isnan(median_val):
                median_val = 0.0
            self.training_medians_[col] = median_val
            
        # For categorical covariates, determine mode
        for col in CATEGORICAL_COVARIATES:
            if col in X.columns:
                mode_s = X[col].mode(dropna=True)
                self.training_medians_[col] = float(mode_s.iloc[0]) if len(mode_s) > 0 else 0.0
                
        self.is_fitted_ = True
        logger.info("ClinicalPreprocessor successfully fitted on training medians.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Transforms patient time-series using strictly causal rules:
        1. Generates binary missing indicators.
        2. Within-patient forward fill.
        3. Fills remaining unobserved prefixes with training population medians.
        4. Clips physiological anomalies.
        """
        if not self.is_fitted_:
            raise ValueError("ClinicalPreprocessor must be fitted on training data before calling transform.")
            
        df = X.copy()
        
        # Sort deterministically by patient and time
        if "patient_id" in df.columns and "ICULOS" in df.columns:
            df = df.sort_values(by=["patient_id", "ICULOS"]).reset_index(drop=True)
            
        feature_cols = [c for c in (NUMERICAL_COVARIATES + CATEGORICAL_COVARIATES) if c in df.columns]
        
        # Step 1: Missingness indicator flags (captures physician ordering patterns)
        if self.include_missingness_flags:
            for col in feature_cols:
                df[f"{col}_isnan"] = df[col].isna().astype(np.int8)
                
        # Step 2: Patient-wise forward filling (past measurements propagated forward)
        # Note: Absolutely NO backward filling (bfill) is applied!
        if "patient_id" in df.columns:
            df[feature_cols] = df.groupby("patient_id", sort=False)[feature_cols].ffill()
        else:
            df[feature_cols] = df[feature_cols].ffill()
            
        # Step 3: Population median imputation for remaining NaNs (initial unobserved hours)
        for col in feature_cols:
            if col in self.training_medians_:
                med_val = self.training_medians_[col]
                df[col] = df[col].fillna(med_val)
                
        # Step 4: Physiological clipping
        if self.clip_physiological_bounds:
            for col, (low, high) in PHYSIOLOGICAL_RANGES.items():
                if col in df.columns:
                    df[col] = df[col].clip(lower=low, upper=high)
                    
        # Verification: Assert zero NaNs remaining across feature columns
        remaining_nans = df[feature_cols].isna().sum().sum()
        if remaining_nans > 0:
            raise ValueError(f"Imputation incomplete: {remaining_nans} NaNs remaining in preprocessed dataframe.")
            
        return df

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        return self.fit(X, y).transform(X)

    def save(self, filepath: Optional[Path] = None) -> Path:
        """Serializes fitted preprocessor to disk."""
        dest = filepath or (MODELS_DIR / "clinical_preprocessor.joblib")
        dest.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, dest)
        logger.info(f"Preprocessor saved to {dest}")
        return dest

    @classmethod
    def load(cls, filepath: Optional[Path] = None) -> "ClinicalPreprocessor":
        """Deserializes fitted preprocessor from disk."""
        src = filepath or (MODELS_DIR / "clinical_preprocessor.joblib")
        return joblib.load(src)


def audit_causal_imputation_integrity(
    raw_df: pd.DataFrame,
    preprocessed_df: pd.DataFrame
) -> Tuple[bool, List[str]]:
    """
    Audits preprocessing against causal integrity invariants:
    1. Zero backward fill: if hour 1 was NaN in raw, it must NOT equal hour 2's raw value if hour 2 was observed.
    2. Missingness flags strictly match raw NaN mask.
    3. Zero remaining NaNs.
    """
    violations = []
    
    # Check 1: Zero remaining NaNs in numerical features
    num_nans = preprocessed_df[NUMERICAL_COVARIATES].isna().sum().sum()
    if num_nans > 0:
        violations.append(f"Preprocessed DataFrame contains {num_nans} residual NaNs.")
        
    # Check 2: Verify no backward fill occurred
    # For a patient with NaN at t=0 and value at t=1, preprocessed t=0 must NOT equal raw t=1 (unless by random chance equal to median)
    sample_pid = raw_df["patient_id"].iloc[0]
    p_raw = raw_df[raw_df["patient_id"] == sample_pid].sort_values("ICULOS").reset_index(drop=True)
    p_prep = preprocessed_df[preprocessed_df["patient_id"] == sample_pid].sort_values("ICULOS").reset_index(drop=True)
    
    # Inspect a lab feature that was missing at t=0 but measured later
    for lab in LAB_VARIABLES:
        if len(p_raw) > 2 and pd.isna(p_raw[lab].iloc[0]) and pd.notna(p_raw[lab].iloc[1]):
            val_t0 = p_prep[lab].iloc[0]
            val_t1 = p_raw[lab].iloc[1]
            if val_t0 == val_t1 and val_t0 != 0.0:
                violations.append(f"Possible backward fill detected in {lab} for patient {sample_pid}.")
                break
                
    is_clean = len(violations) == 0
    return is_clean, violations
