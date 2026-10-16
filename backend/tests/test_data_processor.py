"""Unit tests for Review Forensics DataProcessor and Schema Normalization."""

import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processor import DataProcessor, CANONICAL_COLUMNS, assign_category, generate_product_id


class TestDataProcessor(unittest.TestCase):
    def test_combine_text(self):
        # Both title and body
        self.assertEqual(
            DataProcessor._combine_text("Great cooler", "Chilled airflow and quiet fan"),
            "Great cooler. Chilled airflow and quiet fan",
        )
        # Title already has punctuation
        self.assertEqual(
            DataProcessor._combine_text("Super!", "I loved it"),
            "Super! I loved it",
        )
        # Title only
        self.assertEqual(DataProcessor._combine_text("Amazing", None), "Amazing")
        # Body only
        self.assertEqual(DataProcessor._combine_text(None, "Good product"), "Good product")
        # Handled nan strings
        self.assertEqual(DataProcessor._combine_text("nan", "Valid text"), "Valid text")
        # Both empty
        self.assertEqual(DataProcessor._combine_text(None, np.nan), "")

    def test_clean_amazon_chunk(self):
        processor = DataProcessor(raw_dir="data/raw", processed_dir="data/processed")
        test_chunk = pd.DataFrame({
            0: ["2", "1", "99"],  # 2=pos, 1=neg, 99=invalid
            1: ["Great game", "Terrible", "Bad"],
            2: ["Loved every minute.", "Broke immediately.", "Broken"],
        })
        clean_df, dropped = processor.clean_amazon_chunk(test_chunk)
        self.assertEqual(dropped, 1)  # invalid polarity dropped
        self.assertEqual(len(clean_df), 2)
        self.assertEqual(list(clean_df.columns), CANONICAL_COLUMNS)
        self.assertEqual(clean_df.iloc[0]["sentiment"], "positive")
        self.assertEqual(clean_df.iloc[1]["sentiment"], "negative")
        self.assertTrue(pd.isna(clean_df.iloc[0]["product_name"]))
        self.assertTrue(pd.isna(clean_df.iloc[0]["product_id"]))
        self.assertTrue(pd.isna(clean_df.iloc[0]["rating"]))
        self.assertTrue(pd.isna(clean_df.iloc[0]["price"]))
        self.assertTrue(pd.isna(clean_df.iloc[0]["review_date"]))
        self.assertEqual(clean_df.iloc[0]["source"], "Amazon")

    def test_category_assignment(self):
        self.assertEqual(assign_category("Candes 12 L Room Air Cooler"), "Home Appliances")
        self.assertEqual(assign_category("Pigeon Favourite Electric Kettle 1.5L"), "Kitchen & Dining")
        self.assertEqual(assign_category("Mi 5A 32 inch Smart LED TV"), "Electronics & Media")
        self.assertEqual(assign_category("Lakme Eyeconic Deep Black Kajal"), "Beauty & Personal Care")
        self.assertIsNone(assign_category(None))

    def test_product_id_generation(self):
        pid1 = generate_product_id("Candes Air Cooler", "FK")
        pid2 = generate_product_id("Candes Air Cooler", "FK")
        self.assertTrue(pid1.startswith("FK-"))
        self.assertEqual(pid1, pid2)  # Deterministic

    def test_canonical_columns(self):
        expected = [
            "product_id",
            "product_name",
            "category",
            "price",
            "rating",
            "review_text",
            "review_summary",
            "sentiment",
            "review_date",
            "source",
        ]
        self.assertEqual(CANONICAL_COLUMNS, expected)


if __name__ == "__main__":
    unittest.main()
