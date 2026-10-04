"""
Unit Tests for Phase 9: Benchmark Clinical & Statistical Baselines.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from sepsis_ml.baselines import RuleBasedClinicalScorer
from sepsis_ml.config import MODELS_DIR, TABLES_DIR


class TestBaselines(unittest.TestCase):
    def setUp(self):
        self.toy_df = pd.DataFrame({
            "patient_id": ["p1", "p1", "p2", "p2"],
            "ICULOS": [1, 2, 1, 2],
            "Resp": [16.0, 24.0, 18.0, 26.0],
            "SBP": [120.0, 95.0, 110.0, 85.0],
            "DBP": [70.0, 50.0, 65.0, 45.0],
            "MAP": [85.0, 65.0, 80.0, 58.0],
            "HR": [75.0, 115.0, 80.0, 125.0],
            "Temp": [37.0, 38.5, 36.8, 39.2],
            "WBC": [8.0, 14.5, 7.5, 16.0],
        })

    def test_clinical_qsofa_rule(self):
        """Verify qSOFA probability and binary decision outputs."""
        prob, pred = RuleBasedClinicalScorer.predict_qsofa(self.toy_df)
        self.assertEqual(len(prob), 4)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.all(prob >= 0.0) and np.all(prob <= 1.0))
        # Hour 2 has Resp=24 and SBP=95 -> score=2 -> pred=1
        self.assertEqual(pred[1], 1)
        # Hour 1 has Resp=16 and SBP=120 -> score=0 -> pred=0
        self.assertEqual(pred[0], 0)

    def test_clinical_sirs_rule(self):
        """Verify SIRS score calculation."""
        prob, pred = RuleBasedClinicalScorer.predict_sirs(self.toy_df)
        self.assertEqual(len(prob), 4)
        self.assertEqual(len(pred), 4)
        self.assertTrue(np.all(prob >= 0.0) and np.all(prob <= 1.0))
        # Hour 2 has Temp=38.5, HR=115, Resp=24, WBC=14.5 -> score=4/4=1.0 -> pred=1
        self.assertEqual(pred[1], 1)
        self.assertEqual(prob[1], 1.0)

    def test_clinical_msi_rule(self):
        """Verify Modified Shock Index calculation."""
        prob, pred = RuleBasedClinicalScorer.predict_msi(self.toy_df)
        self.assertEqual(len(prob), 4)
        self.assertEqual(len(pred), 4)
        # Hour 2 MSI = 115 / 65 = 1.77 >= 1.3 -> pred=1
        self.assertEqual(pred[1], 1)

    def test_benchmark_artifacts_exist(self):
        """Verify benchmark table and serialized models were created."""
        bench_csv = TABLES_DIR / "benchmark_baselines.csv"
        self.assertTrue(bench_csv.exists(), f"Benchmark CSV missing at {bench_csv}")
        
        df = pd.read_csv(bench_csv)
        self.assertGreaterEqual(len(df), 6, "Expected at least 6 benchmark models")
        required_cols = ["model", "test_utility_norm", "test_auroc", "test_auprc", "optimal_threshold"]
        for c in required_cols:
            self.assertIn(c, df.columns)
            
        # Check saved joblib models
        for model_name in ["Logistic_Regression", "kNN_k15", "Linear_SVM"]:
            m_path = MODELS_DIR / f"{model_name}.joblib"
            self.assertTrue(m_path.exists(), f"Model file {m_path} missing")


if __name__ == "__main__":
    unittest.main()
