"""
Unit Tests for Phase 6: Temporal-Aware Preprocessing and Imputation.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from sepsis_ml.config import (
    PHYSIOLOGICAL_RANGES,
    RAW_DATA_DIR,
    TARGET_COL,
    VITAL_SIGNS,
)
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.preprocessing import (
    ClinicalPreprocessor,
    NUMERICAL_COVARIATES,
    audit_causal_imputation_integrity,
)
from sepsis_ml.splitting import stratified_patient_split


class TestClinicalPreprocessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        files = list(iter_patient_files(RAW_DATA_DIR))
        cls.raw_df = pd.concat([load_patient_psv(f) for f in files], ignore_index=True)
        cls.train_df, cls.val_df, cls.test_df, _ = stratified_patient_split(cls.raw_df)
        
        cls.preprocessor = ClinicalPreprocessor()
        cls.train_prep = cls.preprocessor.fit_transform(cls.train_df)
        cls.test_prep = cls.preprocessor.transform(cls.test_df)

    def test_zero_residual_nans(self):
        """Verify complete imputation with zero residual NaNs across all partitions."""
        train_nans = self.train_prep[NUMERICAL_COVARIATES].isna().sum().sum()
        test_nans = self.test_prep[NUMERICAL_COVARIATES].isna().sum().sum()
        
        self.assertEqual(train_nans, 0, f"Train set has {train_nans} residual NaNs")
        self.assertEqual(test_nans, 0, f"Test set has {test_nans} residual NaNs")

    def test_no_backward_fill(self):
        """Verify that past missing values are NEVER filled from future measurements."""
        is_clean, violations = audit_causal_imputation_integrity(self.train_df, self.train_prep)
        self.assertTrue(is_clean, f"Causal integrity failed: {violations}")

    def test_training_medians_isolation(self):
        """Verify test set values cannot alter training medians (prevention of distribution leakage)."""
        prep1 = ClinicalPreprocessor().fit(self.train_df)
        
        # Create a corrupted test set with extreme values
        corrupted_test = self.test_df.copy()
        corrupted_test["HR"] = 999.0
        
        # Transform corrupted test set
        _ = prep1.transform(corrupted_test)
        
        # Medians must remain unchanged
        prep2 = ClinicalPreprocessor().fit(self.train_df)
        for col in NUMERICAL_COVARIATES:
            self.assertEqual(prep1.training_medians_[col], prep2.training_medians_[col])

    def test_missingness_flags_consistency(self):
        """Verify that _isnan indicator perfectly matches raw NaN locations."""
        for v in VITAL_SIGNS:
            flag_col = f"{v}_isnan"
            raw_was_nan = self.train_df[v].isna().astype(np.int8).values
            prep_flag = self.train_prep[flag_col].values
            np.testing.assert_array_equal(
                raw_was_nan,
                prep_flag,
                err_msg=f"Missingness flag {flag_col} diverged from raw NaN pattern"
            )

    def test_physiological_clipping(self):
        """Verify extreme synthetic values are clipped to physiological ranges."""
        toy_df = pd.DataFrame({
            "patient_id": ["p999", "p999"],
            "ICULOS": [1, 2],
            "HR": [5.0, 450.0],  # Out of range [20, 300]
            "Temp": [10.0, 50.0],  # Out of range [25, 45]
            "Gender": [1, 1],
            "Unit1": [1, 1],
            "Unit2": [0, 0],
            "Age": [50.0, 50.0],
            "HospAdmTime": [-1.0, -1.0],
            "SepsisLabel": [0, 0],
        })
        # Add remaining columns as NaNs
        for col in NUMERICAL_COVARIATES:
            if col not in toy_df.columns:
                toy_df[col] = np.nan
                
        prep = ClinicalPreprocessor()
        prep.fit(self.train_df)
        transformed = prep.transform(toy_df)
        
        self.assertGreaterEqual(transformed["HR"].min(), PHYSIOLOGICAL_RANGES["HR"][0])
        self.assertLessEqual(transformed["HR"].max(), PHYSIOLOGICAL_RANGES["HR"][1])
        self.assertGreaterEqual(transformed["Temp"].min(), PHYSIOLOGICAL_RANGES["Temp"][0])
        self.assertLessEqual(transformed["Temp"].max(), PHYSIOLOGICAL_RANGES["Temp"][1])

    def test_preprocessor_serialization(self):
        """Verify preprocessor can be saved and reloaded with identical behavior."""
        saved_path = self.preprocessor.save()
        loaded = ClinicalPreprocessor.load(saved_path)
        
        self.assertEqual(self.preprocessor.training_medians_, loaded.training_medians_)
        
        t1 = self.preprocessor.transform(self.test_df.head(10))
        t2 = loaded.transform(self.test_df.head(10))
        pd.testing.assert_frame_equal(t1, t2)


if __name__ == "__main__":
    unittest.main()
