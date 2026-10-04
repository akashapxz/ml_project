"""
Unit Tests for Phase 8: Official PhysioNet Evaluation Metrics Engine.
"""

import unittest
import numpy as np
import pandas as pd

from sepsis_ml.config import TARGET_COL
from sepsis_ml.metrics import (
    compute_comprehensive_metrics,
    compute_patient_utility_matrix,
    compute_physionet_utility,
    optimize_utility_threshold,
)


class TestMetricsEngine(unittest.TestCase):
    def setUp(self):
        # Construct synthetic cohort of 2 patients:
        # Patient 1: Septic (onset at hour 20, length 30 hours)
        # Patient 2: Non-septic (length 30 hours)
        p1_hours = 30
        p1_labels = np.zeros(p1_hours, dtype=int)
        p1_labels[14:] = 1  # 6h before onset at t=20
        
        p2_hours = 30
        p2_labels = np.zeros(p2_hours, dtype=int)
        
        self.cohort_df = pd.DataFrame({
            "patient_id": ["p1"] * p1_hours + ["p2"] * p2_hours,
            "ICULOS": list(range(1, p1_hours + 1)) + list(range(1, p2_hours + 1)),
            TARGET_COL: np.concatenate([p1_labels, p2_labels]),
        })

    def test_utility_all_zeros_baseline_is_zero(self):
        """Verify that predicting all zeros produces exact normalized utility of 0.0."""
        eval_df = self.cohort_df.copy()
        eval_df["pred_zeros"] = 0
        
        res = compute_physionet_utility(eval_df, pred_col="pred_zeros")
        self.assertAlmostEqual(res["utility_norm"], 0.0, places=4)

    def test_utility_optimal_oracle_is_one(self):
        """Verify that an oracle classifier achieves exact normalized utility of 1.0."""
        eval_df = self.cohort_df.copy()
        oracle_preds = []
        for pid, group in eval_df.groupby("patient_id", sort=False):
            is_septic = bool((group[TARGET_COL] == 1).any())
            tsepsis = int(np.where(group[TARGET_COL] == 1)[0][0]) + 6 if is_septic else None
            u_mat = compute_patient_utility_matrix(len(group), is_septic, tsepsis)
            best_action = u_mat.argmax(axis=1)
            oracle_preds.extend(best_action)
            
        eval_df["pred_oracle"] = oracle_preds
        res = compute_physionet_utility(eval_df, pred_col="pred_oracle")
        self.assertAlmostEqual(res["utility_norm"], 1.0, places=4)

    def test_utility_false_alarm_penalty(self):
        """Verify constant false alarms underperform the optimal oracle and accrue FP penalties."""
        eval_df = self.cohort_df.copy()
        eval_df["pred_ones"] = 1
        res_ones = compute_physionet_utility(eval_df, pred_col="pred_ones")
        
        # Oracle utility is strictly 1.0, all-ones must be strictly less than 1.0 due to FP penalties on p2
        self.assertLess(res_ones["utility_norm"], 1.0)
        self.assertGreater(res_ones["utility_norm"], 0.0)
        
        # In a realistic cohort with 10 non-septic patients per septic patient:
        p2_group = self.cohort_df[self.cohort_df["patient_id"] == "p2"]
        expanded_records = [self.cohort_df[self.cohort_df["patient_id"] == "p1"]]
        for i in range(10):
            p_neg = p2_group.copy()
            p_neg["patient_id"] = f"neg_{i}"
            expanded_records.append(p_neg)
        expanded_df = pd.concat(expanded_records, ignore_index=True)
        expanded_df["pred_ones"] = 1
        res_imbalanced = compute_physionet_utility(expanded_df, pred_col="pred_ones")
        # In an imbalanced cohort, all-ones utility drops significantly below 0.5
        self.assertLess(res_imbalanced["utility_norm"], 0.5)

    def test_comprehensive_metrics_bounds(self):
        """Verify classification metrics lie within theoretical mathematical intervals."""
        y_true = np.array([0, 0, 0, 0, 1, 1, 0, 1, 0, 0])
        y_prob = np.array([0.1, 0.2, 0.05, 0.8, 0.9, 0.7, 0.3, 0.6, 0.15, 0.4])
        
        metrics = compute_comprehensive_metrics(y_true, y_prob, threshold=0.5)
        
        self.assertGreaterEqual(metrics["auroc"], 0.0)
        self.assertLessEqual(metrics["auroc"], 1.0)
        self.assertGreaterEqual(metrics["auprc"], 0.0)
        self.assertLessEqual(metrics["auprc"], 1.0)
        self.assertGreaterEqual(metrics["sensitivity"], 0.0)
        self.assertLessEqual(metrics["sensitivity"], 1.0)
        self.assertGreaterEqual(metrics["specificity"], 0.0)
        self.assertLessEqual(metrics["specificity"], 1.0)
        self.assertGreaterEqual(metrics["brier_score"], 0.0)
        self.assertLessEqual(metrics["brier_score"], 1.0)

    def test_threshold_optimization_improves_utility(self):
        """Verify threshold optimization finds an operating point >= default 0.5 threshold."""
        eval_df = self.cohort_df.copy()
        # Create noisy probabilities correlated with target
        rng = np.random.default_state = np.random.RandomState(42)
        eval_df["pred_prob"] = np.where(
            eval_df[TARGET_COL] == 1,
            rng.uniform(0.3, 0.8, len(eval_df)),
            rng.uniform(0.01, 0.3, len(eval_df))
        )
        
        best_th, best_u, curve_df = optimize_utility_threshold(eval_df, prob_col="pred_prob", threshold_steps=20)
        
        eval_df["default_pred"] = (eval_df["pred_prob"] >= 0.5).astype(int)
        default_u = compute_physionet_utility(eval_df, pred_col="default_pred")["utility_norm"]
        
        self.assertGreaterEqual(best_u, default_u)
        self.assertEqual(len(curve_df), 20)


if __name__ == "__main__":
    unittest.main()
