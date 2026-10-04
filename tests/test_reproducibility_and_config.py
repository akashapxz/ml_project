"""
Unit Tests for Phase 1: Environment, Configuration, and Reproducibility.
"""

import os
import random
import unittest
import numpy as np

from sepsis_ml.config import (
    ALL_COLUMNS,
    DEMOGRAPHIC_VARIABLES,
    LAB_VARIABLES,
    PHYSIOLOGICAL_RANGES,
    TARGET_COL,
    VITAL_SIGNS,
    ProjectConfig,
)
from sepsis_ml.reproducibility import seed_everything


class TestReproducibilityAndConfig(unittest.TestCase):
    def test_clinical_variables_count(self):
        """Verify exact count of variables as specified in Reyna et al. 2019."""
        self.assertEqual(len(VITAL_SIGNS), 8, f"Expected 8 vital signs, got {len(VITAL_SIGNS)}")
        self.assertEqual(len(LAB_VARIABLES), 26, f"Expected 26 lab analytes, got {len(LAB_VARIABLES)}")
        self.assertEqual(len(DEMOGRAPHIC_VARIABLES), 6, f"Expected 6 demographic variables, got {len(DEMOGRAPHIC_VARIABLES)}")
        self.assertEqual(TARGET_COL, "SepsisLabel")
        self.assertEqual(len(ALL_COLUMNS), 41, f"Expected 41 total columns (40 features + 1 label), got {len(ALL_COLUMNS)}")

    def test_reproducibility_seed_deterministic(self):
        """Verify seed_everything enforces exact repeatability across random and numpy."""
        seed_everything(42)
        val_py1 = [random.random() for _ in range(5)]
        val_np1 = np.random.normal(size=5).tolist()
        
        seed_everything(42)
        val_py2 = [random.random() for _ in range(5)]
        val_np2 = np.random.normal(size=5).tolist()
        
        self.assertEqual(val_py1, val_py2, "Python random generator is not deterministic with seed_everything")
        self.assertEqual(val_np1, val_np2, "NumPy random generator is not deterministic with seed_everything")

    def test_project_config_defaults(self):
        """Verify default partitioning and utility parameters."""
        cfg = ProjectConfig()
        self.assertAlmostEqual(cfg.train_size + cfg.val_size + cfg.test_size, 1.0)
        self.assertEqual(cfg.feature_families, ["F1", "F2", "F3", "F4"])
        self.assertEqual(cfg.seed, 42)
        self.assertEqual(cfg.utility_dt_optimal, -6.0)


if __name__ == "__main__":
    unittest.main()

