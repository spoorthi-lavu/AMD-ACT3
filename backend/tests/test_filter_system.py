"""Automated tests for the filter system across all endpoints:
Overview (/api/summary), Alerts (/api/alerts), and Momentum (/api/momentum).
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app.server import summary, alerts_with_state, momentum_view


class TestFilterSystem(unittest.TestCase):

    def test_summary_category_filter_cascades_products(self):
        """1. /api/summary applies category and platform filters to the products list."""
        s_acc = summary({"mode": "benchmark", "category": "Accessories"})
        self.assertEqual(s_acc["products"], ["Laptop Backpack 30L", "Power Bank 20000mAh"])

        s_kit = summary({"mode": "benchmark", "category": "Kitchen"})
        self.assertEqual(s_kit["products"], ["Non-stick Cookware Set"])

        s_fash = summary({"mode": "benchmark", "category": "Fashion"})
        self.assertEqual(s_fash["products"], ["Cotton Casual Shirt", "Running Shoes Flex"])

    def test_alerts_filtered_by_category(self):
        """2. /api/alerts filters by category properly."""
        # Kitchen alerts (matches planted benchmark alert)
        al_kit, _ = alerts_with_state({"mode": "benchmark", "category": "Kitchen"})
        self.assertGreater(len(al_kit), 0)
        self.assertTrue(all(a["product"] == "Non-stick Cookware Set" for a in al_kit))

        # Categories without alerts return empty list (per requirement 7)
        al_fash, _ = alerts_with_state({"mode": "benchmark", "category": "Fashion"})
        self.assertEqual(len(al_fash), 0)
        self.assertFalse(any(a["product"] == "Non-stick Cookware Set" for a in al_fash))

    def test_alert_evidence_quotes_mention_product(self):
        """Evidence quotes for an alert must be tied to the alert product."""
        alerts, _ = alerts_with_state({"mode": "benchmark", "category": "Kitchen"})
        self.assertGreater(len(alerts), 0)
        for a in alerts:
            self.assertEqual(a["product"], "Non-stick Cookware Set")
            self.assertGreater(len(a["evidence"]), 0)

    def test_momentum_filtered_by_category(self):
        """3. /api/momentum only returns products in the selected category."""
        m_kit = momentum_view({"mode": "benchmark", "category": "Kitchen"})
        self.assertTrue(all(p["product"] == "Non-stick Cookware Set" for p in m_kit))

        m_fash = momentum_view({"mode": "benchmark", "category": "Fashion"})
        self.assertEqual(set(p["product"] for p in m_fash), {"Cotton Casual Shirt", "Running Shoes Flex"})

    def test_empty_results_for_non_matching_filters(self):
        """Non-matching combinations return empty list instead of stale data."""
        al_empty, _ = alerts_with_state({"mode": "benchmark", "category": "Kitchen", "product": "Running Shoes Flex"})
        self.assertEqual(al_empty, [])

        m_empty = momentum_view({"mode": "benchmark", "category": "Kitchen", "product": "Running Shoes Flex"})
        self.assertEqual(m_empty, [])


if __name__ == "__main__":
    unittest.main()
