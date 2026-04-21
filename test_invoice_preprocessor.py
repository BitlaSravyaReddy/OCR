import unittest

from invoice_preprocessor import preprocess_invoice_data


class InvoicePreprocessorTests(unittest.TestCase):
    def test_broken_expense_rows_are_split(self) -> None:
        sample = {
            "rows": [
                {"row_text": "Invoice No. 1001 Dated 03-11-2025"},
                {"row_text": "Description of Goods | HSN/SAC"},
                {"row_text": "Quantity Rate per Amount"},
                {"row_text": "PADDY 1121"},
                {"row_text": "323.00 qt 3,847.49 qt 12,42,738.00"},
                {"row_text": "Loading Labour MARKET FEE 6,466.46 3,553.00"},
                {"row_text": "Total | 13,59,526.03"},
            ],
            "normalized_fields": {"invoice_no": "1001"},
        }

        out = preprocess_invoice_data(sample, "")
        expenses = out["expenses"]
        amounts = sorted(round(float(item["amount"]), 2) for item in expenses)
        self.assertIn(3553.00, amounts)
        self.assertIn(6466.46, amounts)

    def test_fallback_when_rows_missing(self) -> None:
        out = preprocess_invoice_data({}, "Invoice No: 55\nDate: 05/03/2026\nTotal 1,200.00")
        self.assertEqual(out["header"]["Invoice_Number"], "55")
        self.assertEqual(out["header"]["Invoice_Date"], "2026-03-05")
        self.assertTrue(len(out["products"]) >= 1)
        self.assertTrue(len(out["expenses"]) >= 1)


if __name__ == "__main__":
    unittest.main()

