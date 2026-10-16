# Review Forensics — Commerce Intelligence Platform

Review intelligence and forensic anomaly detection powered by real marketplace datasets (Amazon & Flipkart Kaggle datasets), with honest data provenance and an AMD ROCm GPU AI inference architecture.

## Quick Start (Windows / Mac / Linux)

Run the local web dashboard:
```bash
python run.py
```
*(On Windows: double-click `run.bat`). Runs on Python 3.9+ without external server dependencies. Automatically opens http://127.0.0.1:8765/ in your browser.*

---

## Data Pipeline & Seeding

The raw datasets are located in `data/raw/`:
- `data/raw/flipkart_reviews.csv` (`Dataset-SA.csv`): 205,052 reviews
- `data/raw/amazon_reviews.csv` (`train.csv`): 3,600,000 reviews

### 1. Run the Unified Data Processing Pipeline
Normalizes, deduplicates, and standardizes datasets into the canonical 10-field schema and generates data quality audits:
```bash
python backend/data_processor.py
```

### 2. Seed or Re-seed the Local SQLite Database
Loads 205,047 real Flipkart reviews and 50,000 Amazon reviews into `backend/data/rf.db`:
```bash
python backend/data_processor.py --seed-db --db-amazon-sample 50000
```

### 3. Run Rapid Development / Limited Sample
```bash
python backend/data_processor.py --limit 10000
```

### 4. Inspect Raw Datasets & Schemas Only
```bash
python backend/data_processor.py --inspect-only
```

---

## Canonical Schema & Data Provenance

Both datasets are normalized into a canonical 10-field schema:
1. `product_id`: Deterministic product identifier (e.g. `FK-4A8B1C` for Flipkart; null for Amazon)
2. `product_name`: Verified product title (Flipkart only; null for Amazon)
3. `category`: Inferred product category (Home Appliances, Kitchen & Dining, Electronics & Media, etc.)
4. `price`: Numeric price in INR (Flipkart only; null for Amazon)
5. `rating`: 1.0–5.0 star rating (Flipkart only; null for Amazon)
6. `review_text`: Full review body text
7. `review_summary`: Review headline / subject
8. `sentiment`: `positive`, `neutral`, `negative`
9. `review_date`: Review timestamp (null for both datasets; zero fabrication policy)
10. `source`: Platform origin (`Flipkart` or `Amazon`)

---

## Historical Trend Detection & Synthetic Demo Mode

Because the Kaggle Flipkart and Amazon review datasets lack review timestamps, dates are never fabricated.
- **Real Dataset Analytics Mode:** Computes current product health, rating distributions, pricing analytics, and feedback topics from real reviews.
- **Synthetic Trend Demo Mode:** Toggled directly from the dashboard toolbar or Alerts view to demonstrate the two-proportion $z$-test statistical early-warning engine on simulated dated review batches. All synthetic data is clearly segregated and labeled.

---

## AMD GPU AI Inference Architecture (AMD ACT III Hackathon)

The project includes a decoupled AI inference service interface at [backend/app/amd_inference.py](file:///c:/Users/tanis/Downloads/review-forensics%20(1)/review-forensics/backend/app/amd_inference.py):
- **Hardware Target:** AMD Instinct MI300 / AMD Radeon RX 7000 Series (ROCm 6.x)
- **Engine State:** Host CPU (Rule-Based & Statistical Engine) — Ready for AMD ROCm GPU deployment
- **Structured Schema:** `sentiment`, `confidence`, `complaint_category`, `product_issue`, `recommended_action`
- **Decoupling:** AI inference is decoupled from basic statistics, ensuring responsive dashboards without blocking on GPU availability.

---

## Automated Test Suites

Run all unit tests:
```bash
python -m unittest discover backend/tests
```
