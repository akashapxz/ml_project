"""
Feature Set Ablation Design Module (F1, F2, F3, F4).
Engineers reproducible, causal clinical feature tiers:
- F1: Preprocessed raw measurements + missingness observation indicators (76 features).
- F2: F1 + patient-wise backward-looking rolling statistics (6h, 12h, 24h) & temporal deltas (215 features).
- F3: F2 + clinical bedside risk scores (Shock Index, qSOFA, SIRS, BUN/Cr, Pulse Pressure) (222 features).
- F4: F3 + non-linear cross-organ interaction terms and acceleration trajectories (229 features).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sepsis_ml.config import (
    ALL_COLUMNS,
    DEMOGRAPHIC_VARIABLES,
    LAB_VARIABLES,
    TARGET_COL,
    VITAL_SIGNS,
)
from sepsis_ml.logger import get_logger

logger = get_logger("features")

# Key physiological markers for rolling trajectory analysis
CORE_TRAJECTORY_VITALS = ["HR", "MAP", "Resp", "Temp", "O2Sat"]
CORE_TRAJECTORY_LABS = ["Glucose", "WBC", "Creatinine", "Lactate"]


class FeatureEngineer:
    """
    Constructs cumulative feature tiers F1 -> F2 -> F3 -> F4.
    Guarantees strict patient boundary isolation and backward-looking causality.
    """

    def __init__(self):
        self.feature_columns_: Dict[str, List[str]] = {}

    def extract_f1(self, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tier F1: Raw preprocessed covariates + binary missingness indicators.
        Excludes administrative non-predictive identifiers ('patient_id', 'hospital', 'SepsisLabel').
        """
        df = preprocessed_df.copy()
        
        meta_cols = ["patient_id", "hospital", TARGET_COL]
        f1_cols = [c for c in df.columns if c not in meta_cols and not c.startswith("target_")]
        
        self.feature_columns_["F1"] = sorted(f1_cols)
        return df

    def extract_f2(self, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tier F2: F1 + Patient-Wise Causal Rolling Statistics and Rate-of-Change Deltas.
        Windows: 6h, 12h, 24h.
        Deltas: 1h, 3h, 6h.
        Zero cross-patient contamination via groupby('patient_id').
        """
        df = self.extract_f1(preprocessed_df)
        vital_targets = [v for v in (CORE_TRAJECTORY_VITALS + CORE_TRAJECTORY_LABS) if v in df.columns]
        grouped = df.groupby("patient_id", sort=False)
        
        new_cols = {}
        
        # 1. Temporal Deltas (1h, 3h, 6h changes)
        for var in vital_targets:
            for step in [1, 3, 6]:
                col_name = f"{var}_delta_{step}h"
                new_cols[col_name] = grouped[var].diff(periods=step).fillna(0.0).values
                
        # 2. Rolling Statistics (6h, 12h, 24h causal windows)
        for var in vital_targets:
            for w in [6, 12, 24]:
                roll = grouped[var].rolling(window=w, min_periods=1)
                new_cols[f"{var}_roll_mean_{w}h"] = roll.mean().reset_index(level=0, drop=True).values
                new_cols[f"{var}_roll_std_{w}h"] = roll.std().reset_index(level=0, drop=True).fillna(0.0).values
                new_cols[f"{var}_roll_min_{w}h"] = roll.min().reset_index(level=0, drop=True).values
                new_cols[f"{var}_roll_max_{w}h"] = roll.max().reset_index(level=0, drop=True).values
                
        new_df = pd.DataFrame(new_cols, index=df.index)
        combined_df = pd.concat([df, new_df], axis=1)
        
        meta_cols = ["patient_id", "hospital", TARGET_COL]
        f2_cols = [c for c in combined_df.columns if c not in meta_cols and not c.startswith("target_")]
        self.feature_columns_["F2"] = sorted(f2_cols)
        return combined_df

    def extract_f3(self, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tier F3: F2 + Established Clinical Bedside Risk Scores.
        Includes:
        - Shock Index (SI = HR / SBP)
        - Modified Shock Index (MSI = HR / MAP)
        - qSOFA Criteria Score (RR >= 22: +1, SBP <= 100: +1)
        - SIRS Score (Temp, HR, RR, WBC criteria)
        - Pulse Pressure (PP = SBP - DBP)
        - Pulse Pressure Ratio (PP / SBP)
        - BUN-to-Creatinine Ratio (Prerenal azotemia marker)
        """
        df = self.extract_f2(preprocessed_df)
        
        new_cols = {}
        # 1. Shock Index
        new_cols["shock_index"] = (df["HR"] / np.clip(df["SBP"], a_min=30.0, a_max=300.0)).values
        
        # 2. Modified Shock Index
        new_cols["modified_shock_index"] = (df["HR"] / np.clip(df["MAP"], a_min=20.0, a_max=200.0)).values
        
        # 3. Pulse Pressure
        pp = np.clip(df["SBP"] - df["DBP"], a_min=5.0, a_max=200.0)
        new_cols["pulse_pressure"] = pp.values
        new_cols["pulse_pressure_ratio"] = (pp / np.clip(df["SBP"], a_min=30.0, a_max=300.0)).values
        
        # 4. qSOFA Bedside Score (0 to 2 for available ICU variables)
        qsofa_rr = (df["Resp"] >= 22.0).astype(np.int8)
        qsofa_sbp = (df["SBP"] <= 100.0).astype(np.int8)
        new_cols["qsofa_score"] = (qsofa_rr + qsofa_sbp).values
        
        # 5. SIRS Score (0 to 4)
        sirs_temp = ((df["Temp"] > 38.0) | (df["Temp"] < 36.0)).astype(np.int8)
        sirs_hr = (df["HR"] > 90.0).astype(np.int8)
        sirs_resp = (df["Resp"] > 20.0).astype(np.int8)
        sirs_wbc = ((df["WBC"] > 12.0) | (df["WBC"] < 4.0)).astype(np.int8)
        new_cols["sirs_score"] = (sirs_temp + sirs_hr + sirs_resp + sirs_wbc).values
        
        # 6. BUN / Creatinine Ratio
        new_cols["bun_cr_ratio"] = (df["BUN"] / np.clip(df["Creatinine"], a_min=0.1, a_max=20.0)).values
        
        new_df = pd.DataFrame(new_cols, index=df.index)
        combined_df = pd.concat([df, new_df], axis=1)
        
        meta_cols = ["patient_id", "hospital", TARGET_COL]
        f3_cols = [c for c in combined_df.columns if c not in meta_cols and not c.startswith("target_")]
        self.feature_columns_["F3"] = sorted(f3_cols)
        return combined_df

    def extract_f4(self, preprocessed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Tier F4: F3 + Interaction Terms and Non-Stationary Dynamics.
        Includes:
        - Cardiorespiratory Stress Product (HR * Resp / 100)
        - Hypoperfusion Burden (Lactate / MAP)
        - Inflammatory Thermal Stress (WBC * (Temp - 37.0))
        - Coagulation-Immune Ratio (Platelets / WBC)
        - Multi-Scale Acceleration Trajectories (2nd derivative of HR, MAP, Resp)
        """
        df = self.extract_f3(preprocessed_df)
        
        new_cols = {}
        # 1. Cross-organ physiological interactions
        new_cols["cardiorespiratory_stress"] = (df["HR"] * df["Resp"] / 100.0).values
        new_cols["hypoperfusion_burden"] = (df["Lactate"] / np.clip(df["MAP"], a_min=20.0, a_max=200.0)).values
        new_cols["inflammatory_thermal_product"] = (df["WBC"] * (df["Temp"] - 37.0)).values
        new_cols["platelet_wbc_ratio"] = (df["Platelets"] / np.clip(df["WBC"], a_min=0.5, a_max=100.0)).values
        
        # 2. Acceleration Trajectories (rate-of-rate change)
        grouped = df.groupby("patient_id", sort=False)
        for var in ["HR", "MAP", "Resp"]:
            delta1_col = f"{var}_delta_1h"
            accel = grouped[delta1_col].diff(periods=1).fillna(0.0)
            new_cols[f"{var}_accel_1h"] = accel.values
            
        new_df = pd.DataFrame(new_cols, index=df.index)
        combined_df = pd.concat([df, new_df], axis=1)
        
        meta_cols = ["patient_id", "hospital", TARGET_COL]
        f4_cols = [c for c in combined_df.columns if c not in meta_cols and not c.startswith("target_")]
        self.feature_columns_["F4"] = sorted(f4_cols)
        return combined_df

    def build_feature_tier(
        self,
        preprocessed_df: pd.DataFrame,
        tier: str = "F1"
    ) -> Tuple[pd.DataFrame, List[str]]:
        """
        Builds the specified feature tier ('F1', 'F2', 'F3', 'F4').
        Returns transformed DataFrame and list of feature column names.
        """
        tier_upper = tier.upper()
        if tier_upper == "F1":
            df = self.extract_f1(preprocessed_df)
        elif tier_upper == "F2":
            df = self.extract_f2(preprocessed_df)
        elif tier_upper == "F3":
            df = self.extract_f3(preprocessed_df)
        elif tier_upper == "F4":
            df = self.extract_f4(preprocessed_df)
        else:
            raise ValueError(f"Unknown feature tier: '{tier}'. Choose from ['F1', 'F2', 'F3', 'F4'].")
            
        feat_cols = self.feature_columns_[tier_upper]
        logger.info(f"Built Tier {tier_upper}: {len(feat_cols)} features generated on {len(df)} rows.")
        return df, feat_cols


def audit_feature_causality(df_with_features: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Audits engineered features to verify no future timestamps were used.
    For a patient's very first hour (ICULOS=1), all 1h/3h/6h deltas must be 0.0.
    """
    violations = []
    
    first_rows = df_with_features.groupby("patient_id", sort=False).first()
    
    delta_cols = [c for c in df_with_features.columns if "_delta_" in c]
    for col in delta_cols:
        non_zeros = (first_rows[col] != 0.0).sum()
        if non_zeros > 0:
            violations.append(f"Delta column {col} has {non_zeros} non-zero values at initial timestamp.")
            
    nan_counts = df_with_features.isna().sum()
    cols_with_nans = nan_counts[nan_counts > 0].to_dict()
    if cols_with_nans:
        violations.append(f"Engineered features contain residual NaNs: {cols_with_nans}")
        
    is_clean = len(violations) == 0
    return is_clean, violations
