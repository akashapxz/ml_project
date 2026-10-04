"""
Unit Tests for Phase 5: Patient-Level Data Splitting and Leakage Audit.
"""

import unittest
from pathlib import Path
import pandas as pd

from sepsis_ml.config import ProjectConfig, RAW_DATA_DIR, TARGET_COL
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.splitting import (
    audit_split_overlap,
    build_patient_metadata,
    stratified_patient_split,
)


class TestPatientSplitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = list(iter_patient_files(RAW_DATA_DIR))
        cls.df = pd.concat([load_patient_psv(f) for f in files], ignore_index=True)

    def test_zero_patient_overlap(self):
        """Verify strict zero intersection among Train, Validation, and Test patient sets."""
        cfg = ProjectConfig(seed=42)
        train_df, val_df, test_df, splits = stratified_patient_split(self.df, cfg)
        
        train_pids = set(splits["train"])
        val_pids = set(splits["val"])
        test_pids = set(splits["test"])
        
        # Test pairwise disjointness
        self.assertEqual(len(train_pids & val_pids), 0, "Train and Val share patients!")
        self.assertEqual(len(train_pids & test_pids), 0, "Train and Test share patients!")
        self.assertEqual(len(val_pids & test_pids), 0, "Val and Test share patients!")
        
        # Audit function verification
        is_clean, violations = audit_split_overlap(splits["train"], splits["val"], splits["test"])
        self.assertTrue(is_clean, f"Overlap audit failed: {violations}")

    def test_split_proportions(self):
        """Verify exact 70/15/15 patient split proportions."""
        cfg = ProjectConfig(seed=42)
        _, _, _, splits = stratified_patient_split(self.df, cfg)
        
        total_pts = self.df["patient_id"].nunique()
        self.assertEqual(len(splits["train"]), int(round(total_pts * 0.70)))
        self.assertEqual(len(splits["val"]), int(round(total_pts * 0.15)))
        self.assertEqual(len(splits["test"]), int(round(total_pts * 0.15)))
        self.assertEqual(len(splits["train"]) + len(splits["val"]) + len(splits["test"]), total_pts)

    def test_split_reproducibility(self):
        """Verify deterministic reproducibility of splits with identical seed."""
        cfg = ProjectConfig(seed=42)
        _, _, _, split1 = stratified_patient_split(self.df, cfg)
        _, _, _, split2 = stratified_patient_split(self.df, cfg)
        
        self.assertEqual(split1["train"], split2["train"])
        self.assertEqual(split1["val"], split2["val"])
        self.assertEqual(split1["test"], split2["test"])

    def test_prevalence_stratification(self):
        """Verify sepsis prevalence is preserved within reasonable clinical margin across splits."""
        cfg = ProjectConfig(seed=42)
        meta_df = build_patient_metadata(self.df)
        overall_prev = meta_df["has_sepsis"].mean()
        
        _, _, _, splits = stratified_patient_split(self.df, cfg)
        for name in ["train", "val", "test"]:
            sub_meta = meta_df[meta_df["patient_id"].isin(splits[name])]
            split_prev = sub_meta["has_sepsis"].mean()
            # Assert split prevalence is within 5 percentage points of overall
            self.assertAlmostEqual(split_prev, overall_prev, delta=0.06)


if __name__ == "__main__":
    unittest.main()
