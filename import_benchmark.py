import csv
import sqlite3
import hashlib
import json
from pathlib import Path

csv_path = Path("synthetic_reviews.csv")
db_path = Path("backend/data/rf.db")

if not csv_path.exists():
    raise SystemExit(f"CSV not found: {csv_path}")

db_path.parent.mkdir(parents=True, exist_ok=True)
con = sqlite3.connect(db_path)

try:
    con.execute("""
        CREATE TABLE IF NOT EXISTS synthetic_benchmark_reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            review_text TEXT NOT NULL,
            rating REAL,
            sentiment TEXT,
            topic TEXT,
            sarcasm INTEGER,
            negation INTEGER,
            language TEXT,
            rating_text_mismatch INTEGER,
            prompt_injection INTEGER,
            source TEXT,
            row_hash TEXT UNIQUE
        )
    """)

    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {
            "review_text", "rating", "sentiment", "topic",
            "sarcasm", "negation", "language",
            "rating_text_mismatch", "prompt_injection", "source"
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing CSV columns: {sorted(missing)}")

        batch = []
        for row in reader:
            text = (row.get("review_text") or "").strip()
            if not text:
                continue

            digest = hashlib.sha256(
                json.dumps(row, sort_keys=True).encode("utf-8")
            ).hexdigest()

            def flag(value):
                return int(str(value).strip().lower() in {"true", "1", "yes"})

            batch.append((
                text,
                float(row["rating"]) if row.get("rating") else None,
                row.get("sentiment"),
                row.get("topic"),
                flag(row.get("sarcasm")),
                flag(row.get("negation")),
                row.get("language"),
                flag(row.get("rating_text_mismatch")),
                flag(row.get("prompt_injection")),
                row.get("source"),
                digest
            ))

    con.executemany("""
        INSERT OR IGNORE INTO synthetic_benchmark_reviews (
            review_text, rating, sentiment, topic, sarcasm, negation,
            language, rating_text_mismatch, prompt_injection, source, row_hash
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, batch)

    con.commit()

    total = con.execute(
        "SELECT COUNT(*) FROM synthetic_benchmark_reviews"
    ).fetchone()[0]

    print("CSV rows read:", len(batch))
    print("Benchmark rows stored:", total)
    print("Duplicate rows skipped on import:", len(batch) - con.execute(
        "SELECT changes()"
    ).fetchone()[0])
    print("Existing reviews table was not modified.")

finally:
    con.close()
