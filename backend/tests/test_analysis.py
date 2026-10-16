import sys, os, unittest, datetime as dt
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from app import analysis as A
def prep(rows):
    out = []
    for r in rows:
        a = A.analyze(r["text"], r["rating"]); out.append({**r, "date": dt.date.fromisoformat(r["date"]), "topics": a["topics"], "sentiment": a["sentiment"]})
    return out
TODAY = dt.date(2026, 10, 15)
class T(unittest.TestCase):
    def test_mixed_opinion(self):
        a = A.analyze("Sound is excellent but battery drains quickly.", 4)
        self.assertEqual({(x["topic"], x["sent"]) for x in a["topics"]}, {("sound", "pos"), ("battery", "neg")}); self.assertEqual(a["sentiment"], "neutral"); self.assertEqual(a["mismatch"], 1)
    def test_injected_battery_change_detected(self):
        al, _ = A.detect(prep(A.demo_rows(TODAY)), TODAY)
        self.assertTrue(any(a["topic"] == "battery" and a["kind"] == "neg" and a["product"].startswith("AuraBeat") for a in al))
    def test_false_alarm_rate_without_injection(self):
        seeds, hits = range(1, 31), 0
        for s in seeds:
            al, _ = A.detect(prep(A.demo_rows(TODAY, seed=s, inject=False, decline=False)), TODAY)
            hits += any(a["kind"] == "neg" for a in al)
        print(f"\nFalse-alarm runs (negative alerts, no injected change): {hits}/{len(seeds)}"); self.assertLessEqual(hits / len(seeds), .35)
    def test_detection_rate_with_injection(self):
        seeds, ok = range(1, 31), 0
        for s in seeds:
            al, _ = A.detect(prep(A.demo_rows(TODAY, seed=s, decline=False)), TODAY); ok += any(a["topic"] == "battery" and a["kind"] == "neg" for a in al)
        print(f"\nInjected battery change detected: {ok}/{len(seeds)}"); self.assertGreaterEqual(ok / len(seeds), .7)
    def test_small_sample_no_alert(self):
        rows = [r for r in A.demo_rows(TODAY) if r["date"] >= "2026-10-10"][:10]; al, notes = A.detect(prep(rows), TODAY)
        self.assertEqual(al, []); self.assertTrue(notes)
    def test_new_product_insufficient(self):
        m = A.momentum(prep([r for r in A.demo_rows(TODAY) if r["date"] >= "2026-10-01"][:12]), TODAY); self.assertTrue(all(x["state"] == "Insufficient data" for x in m))
    def test_genuine_decline_classified(self):
        m = {x["product"]: x["state"] for x in A.momentum(prep(A.demo_rows(TODAY)), TODAY)}; print("\nMomentum:", m); self.assertIn(m["AuraFit Earbuds"], ("Emerging decline", "Likely declining"))
if __name__ == "__main__": unittest.main()
