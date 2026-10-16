import os
import sys
import tempfile
import sqlite3
import unittest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_processor import DataProcessor, CANONICAL_COLUMNS
from app import server as S


class TestSeedingRobustness(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.raw_dir = os.path.join(self.temp_dir, "raw")
        self.processed_dir = os.path.join(self.temp_dir, "processed")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)
        self.db_path = os.path.join(self.temp_dir, "test_rf.db")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_csv_seeding_does_not_crash(self):
        # Create 0-byte CSV files
        open(os.path.join(self.raw_dir, "flipkart_reviews.csv"), "w").close()
        open(os.path.join(self.raw_dir, "amazon_reviews.csv"), "w").close()

        dp = DataProcessor(raw_dir=self.raw_dir, processed_dir=self.processed_dir)

        # clean_flipkart should return empty dataframe without raising EmptyDataError
        clean_df, stats = dp.clean_flipkart()
        self.assertTrue(clean_df.empty)
        self.assertEqual(list(clean_df.columns), CANONICAL_COLUMNS)

        # populate_database should execute safely without throwing EmptyDataError
        res = dp.populate_database(db_path=self.db_path)
        self.assertIsNotNone(res)

        # Database tables exist and real reviews count is 0
        conn = sqlite3.connect(self.db_path)
        real_count = conn.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 0").fetchone()[0]
        conn.close()
        self.assertEqual(real_count, 0)

    def test_missing_csv_seeding_does_not_crash(self):
        # Empty directory with no CSVs
        dp = DataProcessor(raw_dir=self.raw_dir, processed_dir=self.processed_dir)
        res = dp.populate_database(db_path=self.db_path)
        self.assertIsNotNone(res)

        conn = sqlite3.connect(self.db_path)
        real_count = conn.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 0").fetchone()[0]
        conn.close()
        self.assertEqual(real_count, 0)

    def test_partial_valid_dataset_ingestion(self):
        # Valid Flipkart CSV with 2 rows, empty Amazon CSV
        fk_content = (
            "product_name,product_price,Rate,Review,Summary,Sentiment\n"
            "Cooler A,3999,4,Chilling breeze and low noise,Great fan,Positive\n"
            "Heater B,1999,1,Stopped working after one day,Poor quality,Negative\n"
        )

        with open(
            os.path.join(self.raw_dir, "flipkart_reviews.csv"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(fk_content)

        # Empty Amazon CSV
        open(os.path.join(self.raw_dir, "amazon_reviews.csv"), "w").close()

        dp = DataProcessor(
            raw_dir=self.raw_dir,
            processed_dir=self.processed_dir,
        )
        dp.populate_database(db_path=self.db_path)

        conn = sqlite3.connect(self.db_path)
        real_count = conn.execute(
            "SELECT COUNT(*) FROM reviews WHERE is_synthetic = 0"
        ).fetchone()[0]
        rows = conn.execute(
            "SELECT product_name, sentiment, is_synthetic "
            "FROM reviews WHERE is_synthetic = 0"
        ).fetchall()
        conn.close()

        self.assertEqual(real_count, 2)
        self.assertEqual(rows[0][2], 0)

    def test_server_startup_with_empty_real_dataset_preserves_synthetic_flag(self):
        # Set server DB to test database
        old_db = S.DB
        S.DB = self.db_path
        try:
            S.init()
            # Simulate server startup check
            c = S.conn()
            real_count = c.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 0").fetchone()[0]
            demo_count = c.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 1").fetchone()[0]
            c.close()

            if demo_count == 0:
                S.load_synthetic_demo()

            # Verify synthetic demo data is loaded and strictly flagged as is_synthetic = 1
            c = S.conn()
            demo_rows = c.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 1").fetchone()[0]
            real_rows = c.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 0").fetchone()[0]
            c.close()

            self.assertGreater(demo_rows, 0)
            self.assertEqual(real_rows, 0)  # Never conflate synthetic demo data with real reviews
        finally:
            S.DB = old_db


if __name__ == "__main__":
    unittest.main()
