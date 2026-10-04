"""
Unit Tests for Phase 4: Leakage-Safe Target Construction and Horizons.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from sepsis_ml.config import RAW_DATA_DIR, TARGET_COL
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.target import (
    add_multi_horizon_targets,
    audit_target_leakage,
    construct_horizon_target,
    extract_patient_sepsis_metadata,
)


class TestTargetConstruction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = list(iter_patient_files(RAW_DATA_DIR))
        cls.frames = [load_patient_psv(f) for f in files]
        cls.df = pd.concat(cls.frames, ignore_index=True)

    def test_official_6h_target_parity(self):
        """Verify construct_horizon_target with horizon=6 perfectly matches raw SepsisLabel."""
        for patient_df in self.frames[:20]:
            constructed_6h = construct_horizon_target(patient_df, horizon_hours=6)
            raw_target = patient_df[TARGET_COL]
            pd.testing.assert_series_equal(
                constructed_6h.reset_index(drop=True),
                raw_target.reset_index(drop=True),
                check_names=False,
                check_dtype=False,
                obj="Constructed 6h target vs official SepsisLabel"
            )

    def test_multi_horizon_logical_ordering(self):
        """
        Verify that a longer lead time horizon (e.g. 12h) triggers earlier
        or equal to a shorter horizon (e.g. 6h and 3h).
        Mathematically: sum(y_12h) >= sum(y_6h) >= sum(y_3h).
        """
        for patient_df in self.frames:
            meta = extract_patient_sepsis_metadata(patient_df)
            if meta["is_septic"]:
                y_12 = construct_horizon_target(patient_df, horizon_hours=12)
                y_6 = construct_horizon_target(patient_df, horizon_hours=6)
                y_3 = construct_horizon_target(patient_df, horizon_hours=3)
                
                self.assertGreaterEqual(y_12.sum(), y_6.sum())
                self.assertGreaterEqual(y_6.sum(), y_3.sum())

    def test_non_septic_invariance(self):
        """Verify non-septic patients remain zero across all prediction horizons."""
        non_septic_frames = [f for f in self.frames if (f[TARGET_COL] == 0).all()][:15]
        for patient_df in non_septic_frames:
            for h in [3, 6, 9, 12]:
                y_h = construct_horizon_target(patient_df, horizon_hours=h)
                self.assertEqual(y_h.sum(), 0, f"Non-septic patient had non-zero target at horizon {h}")

    def test_step_function_monotonicity(self):
        """Verify no patient exhibits target drop from 1 back to 0."""
        for patient_df in self.frames:
            for h in [3, 6, 9, 12]:
                y = construct_horizon_target(patient_df, horizon_hours=h).values
                if (y == 1).any():
                    first_pos = int(np.where(y == 1)[0][0])
                    self.assertTrue(
                        (y[first_pos:] == 1).all(),
                        f"Target reversion detected at horizon {h}"
                    )

    def test_target_leakage_audit_passes(self):
        """Verify full cohort passes the target leakage audit with zero violations."""
        is_clean, violations = audit_target_leakage(self.df)
        self.assertTrue(is_clean, f"Target audit failed with violations: {violations}")


if __name__ == "__main__":
    unittest.main()
