"""Tests unitaires — extraction entités external validation."""

import unittest


class TestExternalValidationExtractEntities(unittest.TestCase):
    def test_beneficiary_last_token_not_country_ignored(self):
        from services.external_validation.engine import ExternalValidationEngine

        eng = ExternalValidationEngine()
        out = eng._extract_entities(
            {
                "swift": {
                    "entities": {
                        "beneficiary": "MILLIKEN CHEMICAL",
                        "country": "TUNISIA",
                        "currency": "EUR",
                        "lc_number": "X",
                    }
                }
            }
        )
        self.assertEqual(out.get("country"), "TUNISIA")
        self.assertIsNone(out.get("beneficiary_country"))


if __name__ == "__main__":
    unittest.main()

