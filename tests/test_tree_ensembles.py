"""
Unit Tests for Phase 10: Tree Ensembles & Cross-Validation Ablation.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from sepsis_ml.config import MODELS_DIR, TABLES_DIR, TARGET_COL
from sepsis_ml.tree_ensembles import get_tree_models


class TestTreeEnsembles(unittest.TestCase):
    def test_stratified_group_kfold_no_patient_overlap(self):
        """Verify that StratifiedGroupKFold strictly partitions patient IDs without overlap."""
        p_ids = np.repeat([f"p_{i}" for i in range(20)], 10)
        y = np.repeat(np.random.choice([0, 1], size=20, p=[0.8, 0.2]), 10)
        
        sgkf = StratifiedGroupKFold(n_splits=5)
        for fold, (train_idx, val_idx) in enumerate(sgkf.split(p_ids, y, groups=p_ids)):
            train_patients = set(p_ids[train_idx])
            val_patients = set(p_ids[val_idx])
            overlap = train_patients.intersection(val_patients)
            self.assertEqual(len(overlap), 0, f"Patient leakage in fold {fold}: {overlap}")

    def test_tree_models_factory(self):
        """Verify factory builds configured tree ensemble models."""
        models = get_tree_models(scale_pos_weight=25.0, random_state=42)
        self.assertIn("Random_Forest", models)
        self.assertIn("XGBoost", models)
        self.assertIn("Hist_Gradient_Boosting", models)

    def test_ablation_artifacts_exist(self):
        """Verify ablation comparison table was created and has all 12 experiments."""
        ablation_csv = TABLES_DIR / "ablation_feature_tiers.csv"
        self.assertTrue(ablation_csv.exists(), f"Ablation CSV missing at {ablation_csv}")
        
        df = pd.read_csv(ablation_csv)
        self.assertEqual(len(df), 12, f"Expected 12 ablation rows, got {len(df)}")
        for col in ["model", "feature_tier", "oof_auroc", "test_auroc", "test_utility_norm"]:
            self.assertIn(col, df.columns)

    def test_model_artifacts_saved(self):
        """Verify trained model joblib files exist."""
        for tier in ["F1", "F2", "F3", "F4"]:
            for model_name in ["Random_Forest", "XGBoost", "Hist_Gradient_Boosting"]:
                p = MODELS_DIR / f"{model_name}_{tier}.joblib"
                self.assertTrue(p.exists(), f"Model artifact {p} missing")


if __name__ == "__main__":
    unittest.main()
