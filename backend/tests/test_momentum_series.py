import unittest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.server import momentum_view


class TestMomentumSeries(unittest.TestCase):
    def test_momentum_series_structure(self):
        # Fetch momentum for benchmark
        res = momentum_view({"mode": "benchmark", "category": "Kitchen"})
        self.assertEqual(len(res), 1)
        prod = res[0]
        self.assertEqual(prod["product"], "Non-stick Cookware Set")
        self.assertTrue(prod["has_enough_data"])

        # Check KPI fields
        self.assertIn("avg_rating", prod)
        self.assertIn("avg_rating_prior", prod)
        self.assertIn("negative_share", prod)
        self.assertIn("negative_share_prior", prod)
        self.assertIn("neg_share_delta", prod)
        self.assertIn("neg_trend", prod)
        self.assertIn("top_complaint_topic", prod)
        self.assertIn("state", prod)
        self.assertIn("early_warning", prod)

        # Check weekly series: exactly 8 weeks W1..W8
        series = prod["weekly_series"]
        self.assertEqual(len(series), 8)
        for idx, wk in enumerate(series):
            self.assertEqual(wk["week_label"], f"W{idx+1}")
            self.assertIn("n_reviews", wk)
            self.assertIn("avg_rating", wk)
            self.assertIn("positive_pct", wk)
            self.assertIn("negative_pct", wk)
            if wk["n_reviews"] == 0:
                self.assertIsNone(wk["positive_pct"])
                self.assertIsNone(wk["negative_pct"])
                self.assertIsNone(wk["avg_rating"])
            else:
                self.assertIsNotNone(wk["positive_pct"])
                self.assertIsNotNone(wk["negative_pct"])

        prior_neg = [w["negative_pct"] for w in series[:4] if w["negative_pct"] is not None]
        last_neg = [w["negative_pct"] for w in series[4:] if w["negative_pct"] is not None]
        avg_prior_neg = sum(prior_neg) / len(prior_neg)
        avg_last_neg = sum(last_neg) / len(last_neg)
        # Check planted spike in Non-stick Cookware Set: sharp rise within W5-W8
        w5_neg = series[4]["negative_pct"]
        w8_neg = series[7]["negative_pct"]
        self.assertGreater(w8_neg, w5_neg + 20.0)
        self.assertGreater(avg_last_neg, avg_prior_neg)

        # Check complaint topics: top 5 topics with last and prior counts
        topics = prod["complaint_topics"]
        self.assertLessEqual(len(topics), 5)
        for t in topics:
            self.assertIn("topic", t)
            self.assertIn("label", t)
            self.assertIn("count_last", t)
            self.assertIn("count_prior", t)

        # Early warning should be non-empty for declining cookware
        self.assertIn(prod["state"], ("Likely declining", "Emerging decline"))
        self.assertIsNotNone(prod["early_warning"])
        self.assertIn("complaints rose", prod["early_warning"])


if __name__ == "__main__":
    unittest.main()
