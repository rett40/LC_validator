"""Tests schéma features risk ML (alignement scaler / inférence)."""

import unittest

from services.risk_service.risk_feature_schema import (
    FEATURE_COLS_CANONICAL,
    FEATURE_COLS_LEGACY_20,
    resolve_matrix_feature_columns,
)


class TestResolveMatrixColumns(unittest.TestCase):
    def test_scaler_20_matches_legacy(self):
        meta = {"feature_cols": FEATURE_COLS_LEGACY_20 + ["extra"]}
        cols = resolve_matrix_feature_columns(meta, 20)
        self.assertEqual(len(cols), 20)
        self.assertEqual(cols, FEATURE_COLS_LEGACY_20)
        self.assertNotIn("ucp_decision_reject", cols)

    def test_metadata_aligned_with_scaler(self):
        meta = {"feature_cols": list(FEATURE_COLS_LEGACY_20)}
        cols = resolve_matrix_feature_columns(meta, 20)
        self.assertEqual(cols, FEATURE_COLS_LEGACY_20)

    def test_canonical_length(self):
        self.assertEqual(len(FEATURE_COLS_CANONICAL), 21)
        self.assertIn("ucp_decision_reject", FEATURE_COLS_CANONICAL)


if __name__ == "__main__":
    unittest.main()
