"""Review Forensics API + Static File Server.
Standard library only (http.server + sqlite3).
Provides high-performance SQL-aggregated endpoints for multi-source eCommerce intelligence
(Flipkart, Amazon, and Synthetic Trend Demo).
"""

import os
import sys
import json
import csv
import io
import re
import sqlite3
import hashlib
import datetime as dt
import mimetypes
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from . import analysis as A
from .amd_inference import amd_engine

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DB = os.environ.get("RF_DB", os.path.join(ROOT, "backend", "data", "rf.db"))
WEB = os.path.join(ROOT, "frontend")
MAX_BYTES = 5_000_000
today = lambda: dt.date.today()


def conn():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    return c


def init():
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    c = conn()
    cols = [r[1] for r in c.execute("PRAGMA table_info(reviews)")]
    if cols and "is_synthetic" not in cols:
        c.execute("DROP TABLE reviews")
    c.executescript("""
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
    c.commit()
    c.close()


def audit(c, what):
    c.execute(
        "INSERT INTO audit(at, what) VALUES(?, ?)",
        (dt.datetime.now().isoformat(timespec="seconds"), what),
    )


def load_synthetic_demo():
    """Seeds synthetic demo rows into the database with is_synthetic = 1."""
    c = conn()
    c.execute("DELETE FROM reviews WHERE is_synthetic = 1")
    t = today()
    demo_data = A.demo_rows(t)
    batch = []
    for r in demo_data:
        h = hashlib.sha1(f"demo|{r['product']}|{r['text']}".encode()).hexdigest()
        a = A.analyze(r["text"], r["rating"])
        batch.append((
            "DEMO-01",
            r["product"],
            "Electronics & Media",
            2499.0,
            float(r["rating"]),
            r["text"],
            r["text"][:30],
            a["sentiment"],
            r["date"],
            r["platform"],
            json.dumps(a["topics"]),
            a["mismatch"],
            a["conf"],
            1,  # is_synthetic = 1
            h,
        ))
    c.executemany("""
    INSERT OR IGNORE INTO reviews (
        product_id, product_name, category, price, rating,
        review_text, review_summary, sentiment, review_date,
        source, topics, mismatch, conf, is_synthetic, h
    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, batch)
    audit(c, f"Loaded {len(batch)} synthetic trend demo reviews")
    c.commit()
    c.close()
    return {"loaded": len(batch), "mode": "demo"}


def summary(q):
    if q.get('mode') == 'benchmark':
        return benchmark_summary(q)
    return _orig_summary(q)


def _orig_summary(q):
    """High-performance SQL aggregated summary responding to source, category,

    and product filters.
    """
    c = conn()
    mode = q.get("mode", "real")
    is_synth = 1 if mode == "demo" else 0

    where_clauses = ["is_synthetic = ?"]
    params: list[Any] = [is_synth]

    # Filters
    platform = q.get("platform") or q.get("source") or ""
    if platform:
        where_clauses.append("source = ?")
        params.append(platform)

    category = q.get("category", "")
    if category:
        where_clauses.append("category = ?")
        params.append(category)

    product = q.get("product", "")
    if product:
        where_clauses.append("product_name = ?")
        params.append(product)

    where_sql = " AND ".join(where_clauses)

    # 1. Overall volume, average rating, price metrics
    row_stats = c.execute(
        f"""
        SELECT 
            COUNT(*) as total,
            ROUND(AVG(rating), 2) as avg_rating,
            COUNT(rating) as rated_count,
            ROUND(AVG(price), 2) as avg_price,
            ROUND(MIN(price), 2) as min_price,
            ROUND(MAX(price), 2) as max_price
        FROM reviews WHERE {where_sql}
        """,
        params,
    ).fetchone()

    total = row_stats["total"]
    avg_rating = row_stats["avg_rating"]

    # 2. Rating breakdown (1 to 5 stars)
    ratings_dict = {str(i): 0 for i in range(1, 6)}
    for r in c.execute(
        f"""
        SELECT CAST(ROUND(rating) AS INT) as star, COUNT(*) as cnt
        FROM reviews WHERE {where_sql} AND rating IS NOT NULL
        GROUP BY star
        """,
        params,
    ):
        if r["star"] and 1 <= r["star"] <= 5:
            ratings_dict[str(r["star"])] = r["cnt"]

    # 3. Sentiment breakdown
    sentiment_dict = {"positive": 0, "neutral": 0, "negative": 0}
    for r in c.execute(
        f"""
        SELECT sentiment, COUNT(*) as cnt
        FROM reviews WHERE {where_sql} AND sentiment IS NOT NULL
        GROUP BY sentiment
        """,
        params,
    ):
        if r["sentiment"] in sentiment_dict:
            sentiment_dict[r["sentiment"]] = r["cnt"]

    # 4. Platform comparison
    platforms_list = []
    for r in c.execute(
        f"""
        SELECT 
            source as platform,
            COUNT(*) as n,
            ROUND(AVG(rating), 2) as avg,
            ROUND(AVG(price), 2) as avg_price
        FROM reviews WHERE {where_sql}
        GROUP BY source
        ORDER BY n DESC
        """,
        params,
    ):
        platforms_list.append({
            "platform": r["platform"],
            "n": r["n"],
            "avg": r["avg"],
            "avg_price": r["avg_price"],
        })

    # 5. Category breakdown
    categories_list = []
    for r in c.execute(
        f"""
        SELECT 
            category,
            COUNT(*) as n,
            ROUND(AVG(rating), 2) as avg,
            ROUND(AVG(price), 2) as avg_price
        FROM reviews WHERE {where_sql} AND category IS NOT NULL
        GROUP BY category
        ORDER BY n DESC
        """,
        params,
    ):
        categories_list.append({
            "category": r["category"],
            "count": r["n"],
            "avg_rating": r["avg"],
            "avg_price": r["avg_price"],
        })

    # 6. Top topics from sample
    topics = {}
    sample_topic_rows = c.execute(
        f"SELECT topics FROM reviews WHERE {where_sql} AND topics != '[]' LIMIT 2000",
        params,
    ).fetchall()
    for row in sample_topic_rows:
        try:
            t_list = json.loads(row["topics"] or "[]")
            for t_item in t_list:
                tp = t_item.get("topic")
                snt = t_item.get("sent")
                if tp and snt in ("pos", "neg"):
                    entry = topics.setdefault(tp, {"pos": 0, "neg": 0})
                    entry[snt] += 1
        except Exception:
            pass

    # 7. Available filter options
    products = [
        r["product_name"]
        for r in c.execute(
            f"SELECT DISTINCT product_name FROM reviews WHERE is_synthetic = ? AND product_name IS NOT NULL ORDER BY product_name LIMIT 300",
            (is_synth,),
        )
    ]
    all_categories = [
        r["category"]
        for r in c.execute(
            f"SELECT DISTINCT category FROM reviews WHERE is_synthetic = ? AND category IS NOT NULL ORDER BY category",
            (is_synth,),
        )
    ]
    all_platforms = [
        r["source"]
        for r in c.execute(
            f"SELECT DISTINCT source FROM reviews WHERE is_synthetic = ? ORDER BY source",
            (is_synth,),
        )
    ]

    # 8. Check for active alerts (demo mode or dated data)
    al, notes = [], []
    if mode == "demo":
        synth_rows = []
        for r in c.execute("SELECT * FROM reviews WHERE is_synthetic = 1"):
            d = dict(r)
            d["date"] = dt.date.fromisoformat(d["review_date"]) if d["review_date"] else None
            d["topics"] = json.loads(d["topics"] or "[]")
            d["product"] = d["product_name"]
            d["text"] = d["review_text"]
            d["platform"] = d["source"]
            synth_rows.append(d)
        al, notes = A.detect(synth_rows, today())
    else:
        notes.append(
            "Historical trend detection requires dated records. The imported Flipkart and Amazon Kaggle datasets "
            "do not contain review timestamps. Use the 'Synthetic Trend Demo' mode to preview the anomaly detection engine."
        )

    # 9. Data quality metrics from database
    quality_rows = [dict(r) for r in c.execute("SELECT * FROM data_quality")]
    c.close()

    return {
        "mode": mode,
        "total": total,
        "avg": avg_rating,
        "rated_count": row_stats["rated_count"],
        "price_stats": {
            "avg": row_stats["avg_price"],
            "min": row_stats["min_price"],
            "max": row_stats["max_price"],
        },
        "ratings": ratings_dict,
        "sentiment": sentiment_dict,
        "platforms": platforms_list,
        "categories": categories_list,
        "topics": topics,
        "active_alerts": len(al),
        "notes": notes,
        "products": products,
        "all_categories": all_categories,
        "all_platforms": all_platforms,
        "data_quality": quality_rows,
        "amd_status": amd_engine.status(),
    }


def reviews(q):
    if q.get('mode') == 'benchmark':
        return benchmark_reviews(q)
    return _orig_reviews(q)


def _orig_reviews(q):
    """Paginated review explorer querying the indexed SQLite database."""
    c = conn()
    mode = q.get("mode", "real")
    is_synth = 1 if mode == "demo" else 0

    where_clauses = ["is_synthetic = ?"]
    params: list[Any] = [is_synth]

    # Filters
    platform = q.get("platform") or q.get("source") or ""
    if platform:
        where_clauses.append("source = ?")
        params.append(platform)

    category = q.get("category", "")
    if category:
        where_clauses.append("category = ?")
        params.append(category)

    product = q.get("product", "")
    if product:
        where_clauses.append("product_name = ?")
        params.append(product)

    sentiment = q.get("sentiment", "")
    if sentiment:
        where_clauses.append("sentiment = ?")
        params.append(sentiment)

    rating = q.get("rating", "")
    if rating:
        where_clauses.append("CAST(ROUND(rating) AS INT) = ?")
        params.append(int(rating))

    txt = q.get("q", "").strip()
    if txt:
        where_clauses.append("review_text LIKE ?")
        params.append(f"%{txt}%")

    where_sql = " AND ".join(where_clauses)

    # Count total matching rows
    total = c.execute(f"SELECT COUNT(*) FROM reviews WHERE {where_sql}", params).fetchone()[0]

    # Pagination
    page = max(1, int(q.get("page", 1)))
    page_size = 15
    offset = (page - 1) * page_size

    # Order
    order_col = "id DESC"
    sort_key = q.get("sort", "")
    if sort_key == "rating":
        order_col = "rating DESC, id DESC"
    elif sort_key == "price":
        order_col = "price DESC, id DESC"

    query_sql = f"""
    SELECT 
        id, product_id, product_name, category, price, rating,
        review_text, review_summary, sentiment, review_date,
        source, topics, mismatch, conf, corrected, is_synthetic
    FROM reviews
    WHERE {where_sql}
    ORDER BY {order_col}
    LIMIT ? OFFSET ?
    """
    rows = []
    for r in c.execute(query_sql, params + [page_size, offset]):
        d = dict(r)
        d["topics"] = json.loads(d["topics"] or "[]")
        d["platform"] = d["source"]
        d["product"] = d["product_name"] or "Unspecified Product (Amazon Polarity Dataset)"
        d["text"] = d["review_text"]
        d["date"] = d["review_date"] or "No Date (Kaggle Dataset)"
        rows.append(d)

    c.close()
    return {
        "total": total,
        "page": page,
        "pages": max(1, -(-total // page_size)),
        "rows": rows,
    }


def alerts_with_state(q):
    if q.get('mode') == 'benchmark':
        return benchmark_alerts_with_state(q)
    return _orig_alerts_with_state(q)


def _orig_alerts_with_state(q):
    """Returns anomaly alerts if dated data / demo mode is active;

    otherwise returns honest disclaimer explaining lack of dates in source data.
    """
    mode = q.get("mode", "real")
    if mode == "demo":
        c = conn()
        synth_rows = []
        for r in c.execute("SELECT * FROM reviews WHERE is_synthetic = 1"):
            d = dict(r)
            d["date"] = dt.date.fromisoformat(d["review_date"]) if d["review_date"] else None
            d["topics"] = json.loads(d["topics"] or "[]")
            d["product"] = d["product_name"]
            d["text"] = d["review_text"]
            d["platform"] = d["source"]
            synth_rows.append(d)
        c.close()
        al, notes = A.detect(synth_rows, today())
        return al, notes

    # Real data mode: Kaggle datasets have no review dates
    return [], [
        "Historical trend detection requires dated records. The imported Flipkart and Amazon Kaggle datasets "
        "do not contain review timestamps. Use the 'Synthetic Trend Demo' mode to preview the anomaly detection engine."
    ]


def momentum_view(q):
    if q.get('mode') == 'benchmark':
        return benchmark_momentum_view(q)
    return _orig_momentum_view(q)


def _orig_momentum_view(q):
    """Returns product momentum metrics for dated records or explains requirement."""
    mode = q.get("mode", "real")
    if mode == "demo":
        c = conn()
        synth_rows = []
        for r in c.execute("SELECT * FROM reviews WHERE is_synthetic = 1"):
            d = dict(r)
            d["date"] = dt.date.fromisoformat(d["review_date"]) if d["review_date"] else None
            d["topics"] = json.loads(d["topics"] or "[]")
            d["product"] = d["product_name"]
            d["text"] = d["review_text"]
            d["platform"] = d["source"]
            synth_rows.append(d)
        c.close()
        return A.momentum(synth_rows, today())

    return [{
        "product": "Marketplace Catalog",
        "state": "Requires Dated Records",
        "indicators": [
            {"name": "Date Availability", "now": "Not Present in Kaggle CSVs", "before": "Historical Window Needed", "signal": "info"}
        ],
        "n": [0, 0],
    }]


BENCH_NOTE = "Locally generated synthetic benchmark data (template-based). Not real customer reviews; labels are not independently verified."
BENCH_NO_TIME = "Review dates are simulated and the alert pattern was planted to test the detector. These are not real trends."


def _bench_where(q):
    where, params = ["1=1"], []
    plat = q.get("platform") or q.get("source") or ""
    for col, val in (("platform", plat), ("category", q.get("category") or ""),
                     ("product_name", q.get("product") or ""), ("sentiment", q.get("sentiment") or "")):
        if val:
            where.append(col + " = ?"); params.append(val)
    try:
        rv = int(q.get("rating") or 0)
    except ValueError:
        rv = 0
    if rv:
        where.append("CAST(ROUND(rating) AS INT) = ?"); params.append(rv)
    txt = (q.get("q") or "").strip()
    if txt:
        where.append("review_text LIKE ?"); params.append("%" + txt + "%")
    return " AND ".join(where), params


def benchmark_summary(q):
    T = "synthetic_benchmark_reviews"
    w, params = _bench_where(q)
    c = conn()
    st = c.execute(f"SELECT COUNT(*) n, ROUND(AVG(rating),2) a, COUNT(rating) rc, ROUND(AVG(price),2) ap, ROUND(MIN(price),2) mn, ROUND(MAX(price),2) mx FROM {T} WHERE {w}", params).fetchone()
    ratings = {str(i): 0 for i in range(1, 6)}
    for r in c.execute(f"SELECT CAST(ROUND(rating) AS INT) star, COUNT(*) n FROM {T} WHERE {w} AND rating IS NOT NULL GROUP BY star", params):
        if r["star"] and 1 <= r["star"] <= 5:
            ratings[str(r["star"])] = r["n"]
    sentiment = {"positive": 0, "neutral": 0, "negative": 0}
    for r in c.execute(f"SELECT sentiment, COUNT(*) n FROM {T} WHERE {w} GROUP BY sentiment", params):
        if r["sentiment"] in sentiment:
            sentiment[r["sentiment"]] = r["n"]
    topics = {}
    for r in c.execute(f"SELECT topic, sentiment, COUNT(*) n FROM {T} WHERE {w} GROUP BY topic, sentiment", params):
        e = topics.setdefault(r["topic"] or "unspecified", {"pos": 0, "neg": 0})
        if r["sentiment"] == "positive":
            e["pos"] += r["n"]
        elif r["sentiment"] == "negative":
            e["neg"] += r["n"]
    categories = [{"category": r["category"], "count": r["n"], "avg_rating": r["a"], "avg_price": r["ap"]}
                  for r in c.execute(f"SELECT category, COUNT(*) n, ROUND(AVG(rating),2) a, ROUND(AVG(price),2) ap FROM {T} WHERE {w} AND category IS NOT NULL GROUP BY category ORDER BY n DESC", params)]
    platforms = [{"platform": r["platform"], "n": r["n"], "avg": r["a"], "avg_price": r["ap"]}
                 for r in c.execute(f"SELECT platform, COUNT(*) n, ROUND(AVG(rating),2) a, ROUND(AVG(price),2) ap FROM {T} WHERE {w} GROUP BY platform ORDER BY n DESC", params)]
    edge = c.execute(f"SELECT SUM(sarcasm) sarcasm, SUM(negation) negation, SUM(rating_text_mismatch) rating_text_mismatch, SUM(prompt_injection) prompt_injection FROM {T} WHERE {w}", params).fetchone()
    scenarios = {r["scenario_type"]: r["n"] for r in c.execute(f"SELECT scenario_type, COUNT(*) n FROM {T} WHERE {w} GROUP BY scenario_type", params)}
    products = [r[0] for r in c.execute(f"SELECT DISTINCT product_name FROM {T} WHERE product_name IS NOT NULL ORDER BY product_name")]
    all_categories = [r[0] for r in c.execute(f"SELECT DISTINCT category FROM {T} WHERE category IS NOT NULL ORDER BY category")]
    all_platforms = [r[0] for r in c.execute(f"SELECT DISTINCT platform FROM {T} WHERE platform IS NOT NULL ORDER BY platform")]
    c.close()
    return {
        "mode": "benchmark", "total": st["n"], "avg": st["a"], "rated_count": st["rc"],
        "price_stats": {"avg": st["ap"], "min": st["mn"], "max": st["mx"]},
        "ratings": ratings, "sentiment": sentiment, "platforms": platforms,
        "categories": categories, "topics": topics, "active_alerts": 0,
        "notes": [BENCH_NOTE, BENCH_NO_TIME], "data_note": BENCH_NOTE,
        "edge_cases": {k: (edge[k] or 0) for k in edge.keys()}, "scenarios": scenarios,
        "products": products, "all_categories": all_categories, "all_platforms": all_platforms,
        "data_quality": [], "amd_status": amd_engine.status(),
    }


def benchmark_reviews(q):
    T = "synthetic_benchmark_reviews"
    w, params = _bench_where(q)
    c = conn()
    total = c.execute(f"SELECT COUNT(*) FROM {T} WHERE {w}", params).fetchone()[0]
    try:
        page = max(1, int(q.get("page", 1)))
    except ValueError:
        page = 1
    size = 15
    sk = q.get("sort")
    order = "rating DESC, id DESC" if sk == "rating" else "price DESC, id DESC" if sk == "price" else "id DESC"
    rows = []
    for r in c.execute(f"SELECT * FROM {T} WHERE {w} ORDER BY {order} LIMIT ? OFFSET ?", params + [size, (page - 1) * size]):
        flags = [n for n in ("sarcasm", "negation", "rating_text_mismatch", "prompt_injection") if r[n]]
        rows.append({
            "id": r["id"], "product_id": r["product_id"], "product": r["product_name"], "product_name": r["product_name"],
            "platform": r["platform"], "source": (r["platform"] or "") + " (simulated)",
            "category": r["category"], "topic": r["topic"], "price": r["price"], "rating": r["rating"],
            "review_summary": None, "sentiment": r["sentiment"],
            "review_text": r["review_text"], "text": r["review_text"],
            "review_date": r["review_date"], "date": (r["review_date"] or "") + " (simulated)",
            "scenario_type": r["scenario_type"], "flags": flags, "language": r["language"],
            "topics": [{"topic": r["topic"]}] if r["topic"] else [], "is_synthetic": 1,
        })
    c.close()
    return {"total": total, "page": page, "pages": max(1, -(-total // size)), "rows": rows, "data_note": BENCH_NOTE}


def _bench_detector_rows():
    shift = today() - dt.date(2026, 10, 17)
    c = conn()
    out = []
    for r in c.execute("SELECT * FROM synthetic_benchmark_reviews WHERE review_date IS NOT NULL"):
        d = dict(r)
        d["date"] = dt.date.fromisoformat(d["review_date"]) + shift
        d["review_date"] = d["date"].isoformat()
        sent = {"positive": "pos", "negative": "neg"}.get(d["sentiment"])
        d["topics"] = [{"topic": d["topic"], "sent": sent}] if (d["topic"] and sent) else []
        d["product"] = d["product_name"]
        d["text"] = d["review_text"]
        d["platform"] = d["platform"]
        d["source"] = d["platform"]
        d["mismatch"] = d["rating_text_mismatch"]
        d["conf"] = 0.8
        out.append(d)
    c.close()
    return out


def benchmark_alerts_with_state(q):
    al, notes = A.detect(_bench_detector_rows(), today())
    return al, notes + [BENCH_NO_TIME]


def benchmark_momentum_view(q):
    return A.momentum(_bench_detector_rows(), today())


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, obj, code=200):
        b = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > MAX_BYTES:
            raise ValueError("File too large (limit 5 MB)")
        return self.rfile.read(n).decode("utf-8", "replace")

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path == "/api/summary":
                return self.send(summary(q))
            if u.path == "/api/reviews":
                return self.send(reviews(q))
            if u.path == "/api/alerts":
                a, n = alerts_with_state(q)
                return self.send({"alerts": a, "notes": n})
            if u.path == "/api/momentum":
                return self.send(momentum_view(q))
            if u.path == "/api/amd_status":
                return self.send(amd_engine.status())
            if u.path == "/api/audit":
                c = conn()
                r = [dict(x) for x in c.execute("SELECT * FROM audit ORDER BY id DESC LIMIT 30")]
                c.close()
                return self.send(r)
            if u.path == "/api/quality_report":
                c = conn()
                rows = [dict(r) for r in c.execute("SELECT * FROM data_quality")]
                c.close()
                return self.send(rows)
            if u.path == "/api/template":
                b = b"platform,product,date,rating,price,text\nFlipkart,My Air Cooler,2026-01-15,4,3999,Great airflow and cool breeze.\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/csv")
                self.send_header("Content-Disposition", "attachment; filename=template.csv")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                return self.wfile.write(b)

            # Serve static files from frontend/
            p = os.path.normpath(os.path.join(WEB, "index.html" if u.path == "/" else u.path.lstrip("/")))
            if not p.startswith(WEB) or not os.path.isfile(p):
                return self.send({"error": "not found"}, 404)
            b = open(p, "rb").read()
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(p)[0] or "text/plain")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)
        except Exception as e:
            self.send({"error": str(e)}, 500)

    def do_POST(self):
        u = urlparse(self.path)
        try:
            raw = self.body()
            if u.path == "/api/amd_infer":
                d = json.loads(raw or "{}")
                text = d.get("text", "")
                if not text:
                    return self.send({"error": "Missing 'text' field"}, 400)
                rating = d.get("rating")
                try: rating = float(rating) if rating is not None else None
                except (ValueError, TypeError): rating = None
                res = amd_engine.infer_batch([{"text": text, "rating": rating}])
                return self.send(res[0] if res else {})
            if u.path == "/api/demo":
                return self.send(load_synthetic_demo())
            if u.path == "/api/alert/status":
                d = json.loads(raw or "{}")
                c = conn()
                c.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (d["key"], d["status"]))
                audit(c, f"alert {d['key']} -> {d['status']}")
                c.commit()
                c.close()
                return self.send({"ok": True})
            if u.path == "/api/alert/action":
                d = json.loads(raw or "{}")
                c = conn()
                c.execute(
                    "INSERT INTO actions(key, note, taken, created) VALUES(?,?,?,?)",
                    (d["key"], str(d.get("note", ""))[:1000], d.get("taken", str(today())), dt.datetime.now().isoformat(timespec="seconds")),
                )
                audit(c, f"action recorded on {d['key']}")
                c.commit()
                c.close()
                return self.send({"ok": True})
            if u.path == "/api/review/sentiment":
                d = json.loads(raw or "{}")
                c = conn()
                c.execute("UPDATE reviews SET sentiment = ?, corrected = 1 WHERE id = ?", (d["sentiment"], d["id"]))
                audit(c, f"Corrected review {d['id']} sentiment to {d['sentiment']}")
                c.commit()
                c.close()
                return self.send({"ok": True})
            self.send({"error": "not found"}, 404)
        except Exception as e:
            self.send({"error": str(e)}, 400)


def serve(port=8765):
    init()
    # Check if database has real or demo reviews; if totally empty, seed both
    c = conn()
    real_count = c.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 0").fetchone()[0]
    demo_count = c.execute("SELECT COUNT(*) FROM reviews WHERE is_synthetic = 1").fetchone()[0]
    c.close()

    if demo_count == 0:
        print("[Server] Seeding synthetic demo data for alert testing...")
        load_synthetic_demo()

    if real_count == 0:
        print("[Server] Real review database has 0 records. Checking for local datasets to seed...")
        try:
            try:
                from data_processor import DataProcessor
            except ImportError:
                from ..data_processor import DataProcessor
            dp = DataProcessor()
            dp.populate_database(db_path=DB, amazon_sample_limit=50000)
        except Exception as e:
            print(f"[Server Warning] Automatic real-review database seeding skipped: {e}")

    return ThreadingHTTPServer(("0.0.0.0", port), H)