import unittest
import urllib.request
import urllib.error
import json
import http.client


class TestUnderTheHood(unittest.TestCase):
    BASE = "http://127.0.0.1:8765"

    def test_frontend_nav_and_sections(self):
        req = urllib.request.urlopen(f"{self.BASE}/")
        html = req.read().decode("utf-8")

        # 1. Nav items: Under the hood present, old ones removed from site-nav
        self.assertIn('data-view="under-the-hood">Under the hood<', html)
        self.assertNotIn('data-view="quality">', html)
        self.assertNotIn('data-view="settings">', html)

        # 2. Hero button links to #data
        self.assertIn("switchView('under-the-hood', 'data')", html)

        # 3. Connector cards link to #data and #amd
        self.assertIn("switchView('under-the-hood', 'amd')", html)

        # 4. Anchor menu at the top of Under the hood
        self.assertIn('class="under-hood-nav"', html)
        self.assertIn('scrollToUnderHoodSection(\'data\'', html)
        self.assertIn('scrollToUnderHoodSection(\'amd\'', html)
        self.assertIn('scrollToUnderHoodSection(\'storage\'', html)

        # 5. Three sections in order
        pos_data = html.find('id="section-data"')
        pos_amd = html.find('id="section-amd"')
        pos_storage = html.find('id="section-storage"')

        self.assertNotEqual(pos_data, -1, "Missing section-data")
        self.assertNotEqual(pos_amd, -1, "Missing section-amd")
        self.assertNotEqual(pos_storage, -1, "Missing section-storage")
        self.assertTrue(pos_data < pos_amd < pos_storage, "Sections must appear in order: data -> amd -> storage")

        # 6. Thin rules between sections
        self.assertIn('class="section-divider"', html)

        # 7. Cleaning rules and synthetic disclaimer
        self.assertIn("Cleaning rules", html)
        self.assertIn("Flipkart, shifted rows", html)
        self.assertIn("Synthetic benchmark disclaimer", html)

        # 8. AMD hardware table and classifier
        self.assertIn("Hardware target", html)
        self.assertIn("Try it on a review", html)
        self.assertIn("id=\"btn-run-infer\"", html)

        # 9. Storage section info
        self.assertIn("SQLite at <code>backend/data/rf.db</code>", html)

    def test_direct_link_under_the_hood_serves_html(self):
        req = urllib.request.urlopen(f"{self.BASE}/under-the-hood")
        self.assertEqual(req.status, 200)
        content_type = req.headers.get("Content-Type", "")
        self.assertIn("text/html", content_type)
        html = req.read().decode("utf-8")
        self.assertIn("Under the hood", html)

    def test_redirect_legacy_routes(self):
        # Test /sources redirect to /under-the-hood#data
        class NoRedirectHandler(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        opener = urllib.request.build_opener(NoRedirectHandler)

        for route, target_loc in [
            ("/sources", "/under-the-hood#data"),
            ("/quality", "/under-the-hood#data"),
            ("/amd", "/under-the-hood#amd"),
            ("/settings", "/under-the-hood#amd"),
        ]:
            try:
                opener.open(f"{self.BASE}{route}")
                self.fail(f"Route {route} should return a 302 redirect")
            except urllib.error.HTTPError as err:
                self.assertEqual(err.code, 302, f"Route {route} returned status {err.code}")
                self.assertEqual(err.headers.get("Location"), target_loc)

    def test_backend_endpoints_intact(self):
        # 1. Quality report
        req = urllib.request.urlopen(f"{self.BASE}/api/quality_report")
        report = json.loads(req.read().decode("utf-8"))
        self.assertEqual(len(report), 2)
        sources = {r["source"]: r for r in report}
        self.assertIn("Flipkart", sources)
        self.assertIn("Amazon", sources)
        self.assertEqual(sources["Flipkart"]["valid_rows"], 205047)

        # 2. AMD status
        req = urllib.request.urlopen(f"{self.BASE}/api/amd_status")
        amd = json.loads(req.read().decode("utf-8"))
        self.assertIn("rocm_active", amd)
        self.assertIn("token_factory_active", amd)

        # 3. AMD infer
        data = json.dumps({"text": "Superb battery life and clear sound.", "rating": 5}).encode("utf-8")
        infer_req = urllib.request.Request(
            f"{self.BASE}/api/amd_infer",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(infer_req)
        self.assertEqual(resp.status, 200)
        res_json = json.loads(resp.read().decode("utf-8"))
        self.assertIn("sentiment", res_json)
        self.assertEqual(res_json["sentiment"], "positive")


if __name__ == "__main__":
    unittest.main()
