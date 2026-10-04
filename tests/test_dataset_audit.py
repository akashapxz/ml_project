"""
Unit Tests for Phase 2: Dataset Verification and Data Quality Audit.
"""

import unittest
from pathlib import Path
import pandas as pd

from sepsis_ml.config import ALL_COLUMNS, RAW_DATA_DIR, TARGET_COL
from sepsis_ml.data_loader import iter_patient_files, load_patient_psv
from sepsis_ml.data_quality import audit_single_patient


class TestDatasetVerification(unittest.TestCase):
    def setUp(self):
        self.files = list(iter_patient_files(RAW_DATA_DIR))
        self.assertTrue(len(self.files) > 0, "No raw patient PSV files found in data/raw")

    def test_file_naming_and_format(self):
        """Verify patient files are pipe-delimited with correct column headers."""
        sample_file = self.files[0]
        df = pd.read_csv(sample_file, sep="|")
        
        # Check column headers match exactly
        self.assertEqual(df.columns.tolist(), ALL_COLUMNS, "Columns do not match official challenge specification")
        self.assertIn(TARGET_COL, df.columns)

    def test_single_patient_audit(self):
        """Verify audit_single_patient returns valid structural diagnostics."""
        sample_file = self.files[0]
        metrics = audit_single_patient(sample_file)
        
        self.assertIn("patient_id", metrics)
        self.assertIn("hospital", metrics)
        self.assertGreaterEqual(metrics["num_rows"], 8, "Expected minimum 8 observation hours per patient")
        self.assertIn(metrics["has_sepsis"], [0, 1])
        self.assertTrue(metrics["iculos_monotonic"], "ICULOS must be monotonically increasing")

    def test_patient_psv_loader(self):
        """Verify load_patient_psv correctly appends patient_id and hospital system."""
        sample_file = self.files[0]
        df = load_patient_psv(sample_file)
        
        self.assertIn("patient_id", df.columns)
        self.assertIn("hospital", df.columns)
        self.assertIn(df["hospital"].iloc[0], ["A", "B"])
        self.assertEqual(len(df.columns), len(ALL_COLUMNS) + 2)


if __name__ == "__main__":
    unittest.main()
