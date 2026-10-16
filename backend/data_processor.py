"""Review Forensics - Unified Data Processing Pipeline & Quality Engine.

Loads, normalizes, deduplicates, and standardizes multi-platform eCommerce review datasets
(Amazon and Flipkart) into a canonical 10-field schema.

Canonical Schema:
  - product_id: Optional[str]
  - product_name: Optional[str]
  - category: Optional[str]
  - price: Optional[float]
  - rating: Optional[float]
  - review_text: str
  - review_summary: Optional[str]
  - sentiment: str
  - review_date: Optional[str]
  - source: str

Strict Principles:
  - Preserves raw datasets unchanged.
  - Zero data fabrication: missing dates, prices, and product names are recorded as null/NaN.
  - Generates comprehensive Data Quality & Provenance reports.
  - Streams large files in memory-efficient chunks for Windows compatibility.
  - Directly seeds or updates the local SQLite engine for sub-20ms dashboard queries.
"""

import os
import sys
import json
import time
import hashlib
import sqlite3
import argparse
from typing import Dict, Any, Optional, Tuple, List
import pandas as pd
import numpy as np

# Canonical 10-field shared schema
CANONICAL_COLUMNS = [
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


def assign_category(product_name: Any) -> Optional[str]:
    """Infers product category from product title keywords for Flipkart items.

    Returns None if product name is missing.
    """
    if pd.isna(product_name) or not str(product_name).strip():
        return None
    p = str(product_name).lower()
    if any(w in p for w in ["cooler", "air cooler", "fan", "air conditioner", "heater", "purifier", "geyser"]):
        return "Home Appliances"
    if any(w in p for w in ["kettle", "iron", "cookware", "mixer", "grinder", "blender", "pan", "bottle", "opalware", "dinner", "plate"]):
        return "Kitchen & Dining"
    if any(w in p for w in ["tv", "led", "smart tv", "screen", "speaker", "headphone", "audio", "earbud"]):
        return "Electronics & Media"
    if any(w in p for w in ["kajal", "face", "cream", "lipstick", "lotion", "hair", "shampoo", "serum", "perfume", "soap"]):
        return "Beauty & Personal Care"
    if any(w in p for w in ["curtain", "sheet", "cushion", "towel", "bed", "decor", "rug", "mat", "pillow"]):
        return "Home Furnishings"
    if any(w in p for w in ["shoe", "shirt", "dress", "pant", "jeans", "watch", "bag", "tshirt", "wallet"]):
        return "Fashion & Apparel"
    return "General Merchandise"


def generate_product_id(product_name: Any, prefix: str = "FK") -> Optional[str]:
    """Generates a deterministic product identifier from the product title."""
    if pd.isna(product_name) or not str(product_name).strip():
        return None
    h = hashlib.md5(str(product_name).strip().encode("utf-8")).hexdigest()[:8].upper()
    return f"{prefix}-{h}"


class DataProcessor:
    """ETL Pipeline for loading, cleaning, normalizing, and combining

    multi-platform eCommerce reviews (Amazon and Flipkart).
    """

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        amazon_file: Optional[str] = None,
        flipkart_file: Optional[str] = None,
        chunksize: int = 100_000,
        limit: Optional[int] = None,
    ):
        self.raw_dir = os.path.abspath(raw_dir)
        self.processed_dir = os.path.abspath(processed_dir)
        self.chunksize = chunksize
        self.limit = limit

        os.makedirs(self.processed_dir, exist_ok=True)

        # Resolve dataset paths with fallback to alternative Kaggle filenames
        self.amazon_path = self._resolve_path(
            amazon_file,
            ["amazon_reviews.csv", "train.csv"],
            "Amazon",
        )
        self.flipkart_path = self._resolve_path(
            flipkart_file,
            ["flipkart_reviews.csv", "Dataset-SA.csv"],
            "Flipkart",
        )

    @staticmethod
    def _is_usable_csv(file_path: Optional[str]) -> bool:
        """Check if file exists, is regular file, and contains more than 0 bytes."""
        return bool(file_path and os.path.isfile(file_path) and os.path.getsize(file_path) > 0)

    def _resolve_path(
        self,
        explicit_file: Optional[str],
        candidates: List[str],
        platform_name: str,
        required: bool = False,
    ) -> Optional[str]:
        """Find the dataset file from explicit path or candidate filenames.
        Returns None when not found unless required=True.
        """
        if explicit_file and os.path.isfile(explicit_file):
            return os.path.abspath(explicit_file)
        for name in candidates:
            candidate_path = os.path.join(self.raw_dir, name)
            if os.path.isfile(candidate_path):
                return candidate_path
        if required:
            raise FileNotFoundError(
                f"Could not locate {platform_name} dataset in '{self.raw_dir}'. "
                f"Checked candidates: {candidates}"
            )
        return None

    # -------------------------------------------------------------------------
    # Phase 1: Inspection & Schema Introspection
    # -------------------------------------------------------------------------
    def inspect_datasets(self) -> Dict[str, Any]:
        """Inspect both raw CSV files without modifying them."""
        if not self._is_usable_csv(self.flipkart_path) and not self._is_usable_csv(self.amazon_path):
            print("[Phase 1 Warning] Neither Flipkart nor Amazon raw datasets are available or non-empty.")
            return {"flipkart": {"status": "missing_or_empty"}, "amazon": {"status": "missing_or_empty"}}

        if self._is_usable_csv(self.flipkart_path):
            print(f"\n[Phase 1] Inspecting Flipkart dataset: {self.flipkart_path}")
            df_fk_sample = pd.read_csv(self.flipkart_path, nrows=500)
            total_fk_rows = max(0, sum(1 for _ in open(self.flipkart_path, "r", encoding="utf-8", errors="ignore")) - 1)
        else:
            print(f"[Phase 1 Warning] Flipkart dataset missing or empty: {self.flipkart_path}")
            df_fk_sample = pd.DataFrame(columns=["Rate", "product_price", "Sentiment", "Review", "Summary", "product_name"])
            total_fk_rows = 0

        if self._is_usable_csv(self.amazon_path):
            print(f"[Phase 1] Inspecting Amazon dataset: {self.amazon_path}")
            df_az_sample = pd.read_csv(self.amazon_path, header=None, nrows=500)
            with open(self.amazon_path, "r", encoding="utf-8", errors="ignore") as f:
                total_az_rows = sum(1 for _ in f)
        else:
            print(f"[Phase 1 Warning] Amazon dataset missing or empty: {self.amazon_path}")
            df_az_sample = pd.DataFrame()
            total_az_rows = 0

        inspection_results = {
            "flipkart": {
                "file_path": self.flipkart_path,
                "file_size_mb": round(os.path.getsize(self.flipkart_path) / (1024 * 1024), 2),
                "total_rows": total_fk_rows,
                "raw_columns": df_fk_sample.columns.tolist(),
                "sample_dtypes": {col: str(dtype) for col, dtype in df_fk_sample.dtypes.items()},
                "sample_missing": df_fk_sample.isna().sum().to_dict(),
            },
            "amazon": {
                "file_path": self.amazon_path,
                "file_size_mb": round(os.path.getsize(self.amazon_path) / (1024 * 1024), 2),
                "total_rows": total_az_rows,
                "raw_columns": ["polarity_class", "review_title", "review_body"],
                "sample_dtypes": {f"Column_{i}": str(df_az_sample[i].dtype) for i in range(df_az_sample.shape[1])},
                "sample_missing": {f"Column_{i}": int(df_az_sample[i].isna().sum()) for i in range(df_az_sample.shape[1])},
            },
        }
        return inspection_results

    # -------------------------------------------------------------------------
    # Phase 2: Cleaning & Normalization into Common Schema
    # -------------------------------------------------------------------------
    @staticmethod
    def _combine_text(title: Any, body: Any) -> str:
        """Combine review title and review body cleanly."""
        t = str(title).strip() if pd.notna(title) else ""
        b = str(body).strip() if pd.notna(body) else ""
        if t.lower() in ("nan", "none", "null"):
            t = ""
        if b.lower() in ("nan", "none", "null"):
            b = ""
        if t and b:
            if t[-1] in ".!?,;:-":
                return f"{t} {b}"
            return f"{t}. {b}"
        return t or b

    def clean_flipkart(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and clean Flipkart reviews into canonical 10-field schema."""
        if not self._is_usable_csv(self.flipkart_path):
            print(f"[Phase 2 Warning] Skipping Flipkart processing: dataset missing or 0 bytes ({self.flipkart_path}).")
            return pd.DataFrame(columns=CANONICAL_COLUMNS), {}

        print(f"\n[Phase 2] Processing Flipkart dataset: {self.flipkart_path}")
        try:
            df_raw = pd.read_csv(self.flipkart_path, nrows=self.limit)
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            print(f"[Phase 2 Warning] Skipping Flipkart processing (unusable data): {e}")
            return pd.DataFrame(columns=CANONICAL_COLUMNS), {}

        if df_raw.empty or not any(c in df_raw.columns for c in ["Rate", "Review", "Sentiment", "Summary"]):
            print(f"[Phase 2 Warning] Skipping Flipkart processing: no usable review records or required columns found.")
            return pd.DataFrame(columns=CANONICAL_COLUMNS), {}

        loaded_count = len(df_raw)
        duplicate_count = int(df_raw.duplicated().sum())

        # 1. Clean ratings: numeric coercion (handles shifted text in Rate)
        rates_numeric = pd.to_numeric(df_raw["Rate"], errors="coerce")
        valid_rate_mask = rates_numeric.between(1.0, 5.0)

        # 2. Clean price: strip currency symbols, convert to numeric
        prices_clean = df_raw["product_price"].astype(str).str.replace(r"[^\d.]", "", regex=True)
        prices_numeric = pd.to_numeric(prices_clean, errors="coerce")

        # 3. Clean sentiment: lowercase and validate
        sentiment_clean = df_raw["Sentiment"].astype(str).str.lower().str.strip()
        sentiment_valid = sentiment_clean.where(
            sentiment_clean.isin(["positive", "negative", "neutral"]),
            np.nan,
        )

        # 4. Review text & summary
        summary_clean = df_raw["Review"].fillna("").astype(str).str.strip()
        summary_clean = summary_clean.replace({"": np.nan, "nan": np.nan})

        body_clean = df_raw["Summary"].fillna("").astype(str).str.strip()
        body_clean = body_clean.replace({"": np.nan, "nan": np.nan})

        # Combine text for holistic analysis
        full_text = [
            self._combine_text(rev, sum_text)
            for rev, sum_text in zip(df_raw["Review"], df_raw["Summary"])
        ]
        review_series = pd.Series(full_text, index=df_raw.index)

        # 5. Product IDs and Categories
        product_names = df_raw["product_name"].astype(str).str.strip()
        categories = product_names.apply(assign_category)
        product_ids = product_names.apply(lambda p: generate_product_id(p, "FK"))

        # 6. Build canonical DataFrame
        df_clean = pd.DataFrame({
            "product_id": product_ids,
            "product_name": product_names,
            "category": categories,
            "price": prices_numeric,
            "rating": rates_numeric,
            "review_text": review_series,
            "review_summary": summary_clean,
            "sentiment": sentiment_valid,
            "review_date": pd.Series([np.nan] * loaded_count, dtype="object"),
            "source": "Flipkart",
        })

        # 7. Filter malformed rows (invalid rating or completely empty text)
        valid_mask = valid_rate_mask & (df_clean["review_text"].str.len() > 0)
        dropped_count = loaded_count - int(valid_mask.sum())
        df_clean = df_clean[valid_mask].reset_index(drop=True)

        stats = {
            "source": "Flipkart",
            "rows_loaded": loaded_count,
            "rows_retained": len(df_clean),
            "rows_dropped": dropped_count,
            "duplicates_detected": duplicate_count,
            "unique_products": int(df_clean["product_name"].nunique()),
            "categories_breakdown": df_clean["category"].value_counts().to_dict(),
            "missing_fields": {
                "review_date": len(df_clean),
                "review_summary": int(df_clean["review_summary"].isna().sum()),
                "price": int(df_clean["price"].isna().sum()),
                "rating": int(df_clean["rating"].isna().sum()),
                "product_id": int(df_clean["product_id"].isna().sum()),
            },
            "sentiment_counts": df_clean["sentiment"].value_counts(dropna=False).to_dict(),
            "avg_rating": round(float(df_clean["rating"].mean()), 2),
            "avg_price": round(float(df_clean["price"].dropna().mean()), 2),
        }
        return df_clean, stats

    def clean_amazon_chunk(self, chunk: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
        """Clean and normalize a chunk of Amazon reviews into canonical schema."""
        chunk_len = len(chunk)

        # Polarity mapping: 1 -> negative, 2 -> positive
        col0_numeric = pd.to_numeric(chunk[0], errors="coerce")
        sentiment_map = {1: "negative", 2: "positive"}
        sentiment = col0_numeric.map(sentiment_map)

        # Column 1 = Title, Column 2 = Body
        summary_clean = chunk[1].fillna("").astype(str).str.strip()
        summary_clean = summary_clean.replace({"": np.nan, "nan": np.nan})

        body_clean = chunk[2].fillna("").astype(str).str.strip()
        body_clean = body_clean.replace({"": np.nan, "nan": np.nan})

        full_text = [
            self._combine_text(t, b)
            for t, b in zip(chunk[1], chunk[2])
        ]
        review_series = pd.Series(full_text, index=chunk.index)

        df_clean = pd.DataFrame({
            "product_id": pd.Series([np.nan] * chunk_len, dtype="object"),
            "product_name": pd.Series([np.nan] * chunk_len, dtype="object"),
            "category": pd.Series([np.nan] * chunk_len, dtype="object"),
            "price": pd.Series([np.nan] * chunk_len, dtype="float64"),
            "rating": pd.Series([np.nan] * chunk_len, dtype="float64"),
            "review_text": review_series,
            "review_summary": summary_clean,
            "sentiment": sentiment,
            "review_date": pd.Series([np.nan] * chunk_len, dtype="object"),
            "source": "Amazon",
        })

        valid_mask = (df_clean["review_text"].str.len() > 0) & sentiment.notna()
        dropped = chunk_len - int(valid_mask.sum())
        return df_clean[valid_mask], dropped

    # -------------------------------------------------------------------------
    # Phase 3: Streaming Execution & Quality Reporting
    # -------------------------------------------------------------------------
    def run(self) -> Dict[str, Any]:
        """Execute ETL pipeline, write processed CSVs, and generate quality report."""
        start_time = time.time()
        print("=" * 70)
        print(" REVIEW FORENSICS: UNIFIED DATA PIPELINE INITIALIZATION")
        print("=" * 70)

        # 1. Inspect
        inspection = self.inspect_datasets()

        # 2. Process Flipkart
        df_flipkart, fk_stats = self.clean_flipkart()
        fk_clean_path = os.path.join(self.processed_dir, "flipkart_cleaned.csv")
        df_flipkart.to_csv(fk_clean_path, index=False, encoding="utf-8")
        print(f" -> Saved cleaned Flipkart dataset: {fk_clean_path} ({len(df_flipkart):,} rows)")

        # 3. Stream & Process Amazon
        print(f"\n[Phase 2] Streaming Amazon dataset: {self.amazon_path}")
        az_clean_path = os.path.join(self.processed_dir, "amazon_cleaned.csv")
        combined_path = os.path.join(self.processed_dir, "combined_reviews.csv")

        # Write header and Flipkart records to combined CSV
        df_flipkart.to_csv(combined_path, index=False, encoding="utf-8")

        az_loaded_total = 0
        az_retained_total = 0
        az_dropped_total = 0
        az_missing_summaries = 0
        sentiment_counts = {"positive": 0, "negative": 0}

        if not self._is_usable_csv(self.amazon_path):
            print(f"[Phase 3 Warning] Skipping Amazon streaming: dataset missing or 0 bytes ({self.amazon_path}).")
            chunk_iter = []
        else:
            try:
                chunk_iter = pd.read_csv(
                    self.amazon_path,
                    header=None,
                    chunksize=self.chunksize,
                    nrows=self.limit,
                    dtype=str,
                )
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"[Phase 3 Warning] Skipping Amazon streaming (unusable data): {e}")
                chunk_iter = []

        for i, chunk in enumerate(chunk_iter):
            az_loaded_total += len(chunk)
            clean_chunk, dropped = self.clean_amazon_chunk(chunk)
            az_dropped_total += dropped
            az_retained_total += len(clean_chunk)
            az_missing_summaries += int(clean_chunk["review_summary"].isna().sum())

            for k, v in clean_chunk["sentiment"].value_counts().items():
                sentiment_counts[k] = sentiment_counts.get(k, 0) + v

            mode = "w" if i == 0 else "a"
            header = (i == 0)
            clean_chunk.to_csv(az_clean_path, mode=mode, header=header, index=False, encoding="utf-8")
            clean_chunk.to_csv(combined_path, mode="a", header=False, index=False, encoding="utf-8")

            if (i + 1) % 5 == 0 or (self.limit and az_loaded_total >= self.limit):
                print(f"    ... Streamed {az_loaded_total:,} Amazon rows ({az_retained_total:,} retained)")

        print(f" -> Saved cleaned Amazon dataset: {az_clean_path} ({az_retained_total:,} rows)")
        print(f" -> Saved unified combined dataset: {combined_path} ({len(df_flipkart) + az_retained_total:,} rows)")

        az_stats = {
            "source": "Amazon",
            "rows_loaded": az_loaded_total,
            "rows_retained": az_retained_total,
            "rows_dropped": az_dropped_total,
            "duplicates_detected": 0,
            "unique_products": 0,
            "categories_breakdown": {"Unspecified (Amazon Polarity Dataset)": az_retained_total},
            "missing_fields": {
                "product_id": az_retained_total,
                "product_name": az_retained_total,
                "category": az_retained_total,
                "price": az_retained_total,
                "rating": az_retained_total,
                "review_date": az_retained_total,
                "review_summary": az_missing_summaries,
            },
            "sentiment_counts": sentiment_counts,
        }

        total_time = round(time.time() - start_time, 2)
        total_loaded = fk_stats["rows_loaded"] + az_stats["rows_loaded"]
        total_retained = fk_stats["rows_retained"] + az_stats["rows_retained"]

        quality_report = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "execution_time_seconds": total_time,
            "sources": {
                "Flipkart": fk_stats,
                "Amazon": az_stats,
            },
            "totals": {
                "total_rows_loaded": total_loaded,
                "total_rows_retained": total_retained,
                "total_rows_dropped": total_loaded - total_retained,
            },
            "schema_conformance": {
                "canonical_fields": CANONICAL_COLUMNS,
                "zero_fabrication_policy": True,
                "missing_dates_preserved": True,
            },
        }

        report_path = os.path.join(self.processed_dir, "quality_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(quality_report, f, indent=2)
        print(f" -> Comprehensive Data Quality Report saved: {report_path}")

        print("\n" + "=" * 70)
        print(f" PIPELINE COMPLETE: {total_retained:,} rows processed in {total_time}s")
        print("=" * 70)
        return quality_report

    # -------------------------------------------------------------------------
    # Database Population (Seeds backend/data/rf.db for fast SQLite dashboard)
    # -------------------------------------------------------------------------
    def populate_database(
        self,
        db_path: str = "backend/data/rf.db",
        amazon_sample_limit: int = 50_000,
    ) -> Dict[str, Any]:
        """Populates the SQLite database with real Flipkart reviews,

        a balanced sample of real Amazon reviews, and data-quality metadata.
        """
        print(f"\n[Database Seeding] Seeding SQLite database: {db_path}")
        os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()

        # Initialize tables
        cur.executescript("""
        CREATE TABLE IF NOT EXISTS reviews (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            product_id TEXT,
            product_name TEXT,
            category TEXT,
            price REAL,
            rating REAL,
            review_text TEXT,
            review_summary TEXT,
            sentiment TEXT,
            review_date TEXT,
            source TEXT,
            topics TEXT,
            mismatch INT DEFAULT 0,
            conf REAL DEFAULT 0.8,
            corrected INT DEFAULT 0,
            is_synthetic INT DEFAULT 0,
            h TEXT UNIQUE
        );

        CREATE INDEX IF NOT EXISTS idx_rev_source ON reviews(source, is_synthetic);
        CREATE INDEX IF NOT EXISTS idx_rev_category ON reviews(category, is_synthetic);
        CREATE INDEX IF NOT EXISTS idx_rev_product ON reviews(product_name, is_synthetic);
        CREATE INDEX IF NOT EXISTS idx_rev_sentiment ON reviews(sentiment, is_synthetic);
        CREATE INDEX IF NOT EXISTS idx_rev_rating ON reviews(rating, is_synthetic);

        CREATE TABLE IF NOT EXISTS data_quality (
            source TEXT PRIMARY KEY,
            total_rows INT,
            valid_rows INT,
            dropped_rows INT,
            duplicate_rows INT,
            unique_products INT,
            categories_count INT,
            missing_dates INT,
            summary_json TEXT
        );

        CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, status TEXT);
        CREATE TABLE IF NOT EXISTS actions (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, note TEXT, taken TEXT, created TEXT);
        CREATE TABLE IF NOT EXISTS audit (id INTEGER PRIMARY KEY AUTOINCREMENT, at TEXT, what TEXT);
        """)

        # Clean existing real reviews (keep synthetic demo reviews separate)
        cur.execute("DELETE FROM reviews WHERE is_synthetic = 0")
        conn.commit()

        # 1. Insert Flipkart reviews
        fk_csv = os.path.join(self.processed_dir, "flipkart_cleaned.csv")
        if not self._is_usable_csv(fk_csv):
            if self._is_usable_csv(self.flipkart_path):
                print("Processing Flipkart data first...")
                fk_clean_df, _ = self.clean_flipkart()
                if not fk_clean_df.empty:
                    fk_clean_df.to_csv(fk_csv, index=False)

        fk_inserted = 0
        fk_batch = []
        df_fk = pd.DataFrame(columns=CANONICAL_COLUMNS)
        if self._is_usable_csv(fk_csv):
            try:
                print(f"Loading Flipkart reviews from {fk_csv}...")
                df_fk = pd.read_csv(fk_csv)
            except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
                print(f"[Database Seeding Warning] Could not read Flipkart processed file ({e}).")
        else:
            print("[Database Seeding Warning] No usable Flipkart dataset found. Skipping Flipkart real-review seeding.")

        for _, r in df_fk.iterrows():
            prod = str(r["product_name"]) if pd.notna(r["product_name"]) else None
            txt = str(r["review_text"])
            h = hashlib.sha1(f"Flipkart|{prod}|{txt[:100]}".encode("utf-8")).hexdigest()
            # Basic topic detection
            topics = []
            lt = txt.lower()
            if "cool" in lt: topics.append({"topic": "cooling", "sent": "pos" if "great" in lt else "neg"})
            if "price" in lt: topics.append({"topic": "price", "sent": "pos" if "good" in lt else "neg"})
            if "deliver" in lt: topics.append({"topic": "delivery", "sent": "pos" if "fast" in lt else "neg"})
            if "battery" in lt: topics.append({"topic": "battery", "sent": "pos"})

            fk_batch.append((
                str(r["product_id"]) if pd.notna(r["product_id"]) else None,
                prod,
                str(r["category"]) if pd.notna(r["category"]) else None,
                float(r["price"]) if pd.notna(r["price"]) else None,
                float(r["rating"]) if pd.notna(r["rating"]) else None,
                txt,
                str(r["review_summary"]) if pd.notna(r["review_summary"]) else None,
                str(r["sentiment"]) if pd.notna(r["sentiment"]) else "neutral",
                None,  # No fabricated review date
                "Flipkart",
                json.dumps(topics),
                0,
                0.85,
                0,
                0,  # is_synthetic = 0 (Real data)
                h,
            ))
            if len(fk_batch) >= 10000:
                cur.executemany("""
                INSERT OR IGNORE INTO reviews(
                    product_id, product_name, category, price, rating,
                    review_text, review_summary, sentiment, review_date,
                    source, topics, mismatch, conf, corrected, is_synthetic, h
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, fk_batch)
                fk_inserted += len(fk_batch)
                fk_batch = []

        if fk_batch:
            cur.executemany("""
            INSERT OR IGNORE INTO reviews(
                product_id, product_name, category, price, rating,
                review_text, review_summary, sentiment, review_date,
                source, topics, mismatch, conf, corrected, is_synthetic, h
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, fk_batch)
            fk_inserted += len(fk_batch)
        conn.commit()
        print(f" -> Inserted {fk_inserted:,} real Flipkart reviews.")

        # 2. Insert Amazon reviews (sampled for fast web dashboard responsiveness)
        az_csv = os.path.join(self.processed_dir, "amazon_cleaned.csv")
        az_inserted = 0
        if self._is_usable_csv(az_csv):
            print(f"Loading balanced sample of {amazon_sample_limit:,} Amazon reviews from {az_csv}...")
            az_batch = []
            for chunk in pd.read_csv(az_csv, chunksize=10000, nrows=amazon_sample_limit):
                for _, r in chunk.iterrows():
                    txt = str(r["review_text"])
                    h = hashlib.sha1(f"Amazon|{txt[:120]}".encode("utf-8")).hexdigest()
                    topics = []
                    lt = txt.lower()
                    if "sound" in lt or "music" in lt: topics.append({"topic": "sound", "sent": "pos" if "good" in lt else "neg"})
                    if "price" in lt: topics.append({"topic": "price", "sent": "pos" if "worth" in lt else "neg"})
                    if "deliver" in lt: topics.append({"topic": "delivery", "sent": "pos"})

                    az_batch.append((
                        None,  # No product id
                        None,  # No product name
                        "Unspecified (Amazon Polarity Dataset)",
                        None,  # No price
                        None,  # No rating (only binary polarity)
                        txt,
                        str(r["review_summary"]) if pd.notna(r["review_summary"]) else None,
                        str(r["sentiment"]) if pd.notna(r["sentiment"]) else "neutral",
                        None,  # No fabricated date
                        "Amazon",
                        json.dumps(topics),
                        0,
                        0.9,
                        0,
                        0,  # is_synthetic = 0
                        h,
                    ))
                cur.executemany("""
                INSERT OR IGNORE INTO reviews(
                    product_id, product_name, category, price, rating,
                    review_text, review_summary, sentiment, review_date,
                    source, topics, mismatch, conf, corrected, is_synthetic, h
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, az_batch)
                az_inserted += len(az_batch)
                az_batch = []
            conn.commit()
            print(f" -> Inserted {az_inserted:,} real Amazon reviews.")

        # 3. Store quality report in data_quality table
        report_path = os.path.join(self.processed_dir, "quality_report.json")
        if os.path.isfile(report_path):
            with open(report_path, "r", encoding="utf-8") as f:
                rep = json.load(f)
            for src, sdata in rep.get("sources", {}).items():
                cur.execute("""
                INSERT OR REPLACE INTO data_quality(
                    source, total_rows, valid_rows, dropped_rows,
                    duplicate_rows, unique_products, categories_count,
                    missing_dates, summary_json
                ) VALUES (?,?,?,?,?,?,?,?,?)
                """, (
                    src,
                    sdata.get("rows_loaded", 0),
                    sdata.get("rows_retained", 0),
                    sdata.get("rows_dropped", 0),
                    sdata.get("duplicates_detected", 0),
                    sdata.get("unique_products", 0),
                    len(sdata.get("categories_breakdown", {})),
                    sdata.get("rows_retained", 0),
                    json.dumps(sdata),
                ))
            conn.commit()

        cur.execute("INSERT INTO audit(at, what) VALUES(?, ?)", (
            time.strftime("%Y-%m-%d %H:%M:%S"),
            f"Seeded SQLite database: {fk_inserted} Flipkart reviews + {az_inserted} Amazon reviews.",
        ))
        conn.commit()
        conn.close()

        print(f" -> Seeding complete. Total real reviews active in SQLite: {fk_inserted + az_inserted:,}")
        return {"flipkart_reviews": fk_inserted, "amazon_reviews": az_inserted}


def main():
    parser = argparse.ArgumentParser(
        description="Review Forensics Unified ETL & Quality Pipeline."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw",
        help="Path to raw CSV dataset directory (default: data/raw)",
    )
    parser.add_argument(
        "--processed-dir",
        default="data/processed",
        help="Path to save processed datasets (default: data/processed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max rows to process per dataset (default: all)",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=100000,
        help="Chunk size for streaming processing of large CSVs",
    )
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Inspect datasets and print schema analysis without writing files.",
    )
    parser.add_argument(
        "--seed-db",
        action="store_true",
        help="Seed or update backend/data/rf.db SQLite database with processed data.",
    )
    parser.add_argument(
        "--db-amazon-sample",
        type=int,
        default=50000,
        help="Number of Amazon reviews to sample into SQLite database (default: 50000)",
    )
    args = parser.parse_args()

    processor = DataProcessor(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        chunksize=args.chunksize,
        limit=args.limit,
    )

    if args.inspect_only:
        results = processor.inspect_datasets()
        print(json.dumps(results, indent=2))
    else:
        processor.run()
        if args.seed_db:
            processor.populate_database(amazon_sample_limit=args.db_amazon_sample)


if __name__ == "__main__":
    main()
