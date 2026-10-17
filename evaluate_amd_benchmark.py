import csv
import sqlite3
import time
from pathlib import Path
from collections import Counter
from backend.app.amd_inference import amd_engine

DB = Path("backend/data/rf.db")
OUT = Path("amd_benchmark_eval_20.csv")

with sqlite3.connect(DB) as con:
    rows = con.execute("""
        SELECT id, review_text, rating, sentiment
        FROM synthetic_benchmark_reviews
        ORDER BY id
        LIMIT 20
    """).fetchall()

if not rows:
    raise SystemExit("No benchmark rows found.")

results = []
for i, (row_id, review_text, rating, expected) in enumerate(rows, 1):
    result = amd_engine.infer_batch([{
        "text": review_text,
        "rating": rating
    }])[0]

    engine = result.get("processing_engine", "")
    cloud = engine.startswith("AMD Token Factory")

    results.append({
        "benchmark_id": row_id,
        "rating": rating,
        "expected_sentiment": expected,
        "predicted_sentiment": result.get("sentiment"),
        "confidence": result.get("confidence"),
        "complaint_category": result.get("complaint_category"),
        "processing_engine": engine,
        "cloud_inference": cloud,
    })

    print(
        f"{i}/20 | cloud={cloud} | "
        f"expected={expected} | predicted={result.get('sentiment')}"
    )

    # Avoid sending requests too rapidly to the shared endpoint.
    if i < len(rows):
        time.sleep(2)

with OUT.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

cloud_count = sum(r["cloud_inference"] for r in results)
matches = sum(
    r["expected_sentiment"] == r["predicted_sentiment"]
    for r in results
)

print("\n--- Summary ---")
print("Rows evaluated:", len(results))
print("Cloud responses:", cloud_count)
print("CPU fallback responses:", len(results) - cloud_count)
print("Label matches:", matches, "/", len(results))
print("Raw agreement:", round(matches / len(results), 3))
print("Predicted sentiment counts:",
      dict(Counter(r["predicted_sentiment"] for r in results)))
print("Results saved to:", OUT)
print("Note: agreement is provisional; synthetic labels are not independently validated.")
