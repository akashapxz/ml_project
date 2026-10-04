"""
Unit Tests for Phase 7: Feature Set Ablation Design (F1, F2, F3, F4).
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from sepsis_ml.config import RAW_DATA_DIR, TARGET_COL
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.features import FeatureEngineer, audit_feature_causality
from sepsis_ml.preprocessing import ClinicalPreprocessor
from sepsis_ml.splitting import stratified_patient_split


class TestFeatureEngineering(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = list(iter_patient_files(RAW_DATA_DIR))
        cls.raw_df = pd.concat([load_patient_psv(f) for f in files], ignore_index=True)
        cls.train_df, _, _, _ = stratified_patient_split(cls.raw_df)
        cls.prep = ClinicalPreprocessor()
        cls.train_prep = cls.prep.fit_transform(cls.train_df)
        cls.fe = FeatureEngineer()

    def test_feature_tier_progression(self):
        """Verify strict monotonic subset progression: |F1| < |F2| < |F3| < |F4|."""
        _, f1_cols = self.fe.build_feature_tier(self.train_prep, "F1")
        _, f2_cols = self.fe.build_feature_tier(self.train_prep, "F2")
        _, f3_cols = self.fe.build_feature_tier(self.train_prep, "F3")
        _, f4_cols = self.fe.build_feature_tier(self.train_prep, "F4")
        
        self.assertLess(len(f1_cols), len(f2_cols))
        self.assertLess(len(f2_cols), len(f3_cols))
        self.assertLess(len(f3_cols), len(f4_cols))
        
        # Exact column counts
        self.assertEqual(len(f1_cols), 80)
        self.assertEqual(len(f2_cols), 215)
        self.assertEqual(len(f3_cols), 222)
        self.assertEqual(len(f4_cols), 229)

    def test_zero_future_leakage(self):
        """Verify causality audit passes across F4 dataframe."""
        f4_df, _ = self.fe.build_feature_tier(self.train_prep, "F4")
        is_clean, violations = audit_feature_causality(f4_df)
        self.assertTrue(is_clean, f"Feature causality violated: {violations}")

    def test_patient_boundary_isolation(self):
        """Verify rolling stats and deltas never bleed between two consecutive patients."""
        sample_df = self.train_prep[self.train_prep["patient_id"].isin(self.train_prep["patient_id"].unique()[:2])].copy()
        f2_df, _ = self.fe.build_feature_tier(sample_df, "F2")
        
        first_row_p2 = f2_df.groupby("patient_id").first().iloc[1]
        self.assertEqual(first_row_p2["HR_delta_1h"], 0.0, "Delta bled from Patient 1 to Patient 2")

    def test_clinical_risk_scores_logic(self):
        """Verify Shock Index, qSOFA, and SIRS calculations."""
        toy_df = pd.DataFrame({
            "patient_id": ["p1"],
            "hospital": ["A"],
            "ICULOS": [1],
            "HR": [110.0],
            "SBP": [100.0],
            "DBP": [60.0],
            "MAP": [73.3],
            "Resp": [24.0],
            "Temp": [38.5],
            "WBC": [15.0],
            "BUN": [20.0],
            "Creatinine": [1.0],
            "Lactate": [2.5],
            "Platelets": [250.0],
            "SepsisLabel": [0],
        })
        prep = ClinicalPreprocessor().fit(self.train_df)
        prep_toy = prep.transform(toy_df)
        
        f3_df, _ = self.fe.build_feature_tier(prep_toy, "F3")
        row = f3_df.iloc[0]
        
        # Shock Index = 110 / 100 = 1.10
        self.assertAlmostEqual(row["shock_index"], 1.10, places=2)
        # Pulse Pressure = 100 - 60 = 40
        self.assertEqual(row["pulse_pressure"], 40.0)
        # qSOFA: Resp >= 22 (1) + SBP <= 100 (1) = 2
        self.assertEqual(row["qsofa_score"], 2)
        # SIRS: Temp > 38 (1) + HR > 90 (1) + Resp > 20 (1) + WBC > 12 (1) = 4
        self.assertEqual(row["sirs_score"], 4)
        # BUN / Cr = 20 / 1 = 20.0
        self.assertEqual(row["bun_cr_ratio"], 20.0)


if __name__ == "__main__":
    unittest.main()
