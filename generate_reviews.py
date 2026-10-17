import os
import json
import time
import random
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(override=True)

API_KEY = os.getenv("AMD_API_KEY")
if not API_KEY:
    raise ValueError("AMD_API_KEY missing from .env")

client = OpenAI(
    api_key=API_KEY,
    base_url="https://developer.amd.com.cn/radeon/api/v1",
    timeout=180,
    max_retries=0,
)

MODEL = "DeepSeek-V4-Flash"
OUTPUT = "synthetic_reviews.csv"
TARGET = 10000
BATCH_SIZE = 20

COLUMNS = [
    "review_text", "rating", "sentiment", "topic",
    "sarcasm", "negation", "language",
    "rating_text_mismatch", "prompt_injection", "source",
]

TOPICS = [
    "product_quality", "delivery", "pricing",
    "customer_service", "packaging", "refund",
    "usability", "product_features",
]

BOOLEAN_FIELDS = [
    "sarcasm", "negation", "rating_text_mismatch", "prompt_injection",
]

if os.path.exists(OUTPUT):
    df = pd.read_csv(OUTPUT)
    if not set(COLUMNS).issubset(df.columns):
        raise ValueError(
            f"{OUTPUT} has an unexpected schema. Back it up and inspect it before continuing."
        )
    df = df[COLUMNS].dropna(subset=["review_text"]).copy()
    df["review_text"] = df["review_text"].astype(str).str.strip()
    df = df[df["review_text"] != ""]
    df["_dedup"] = df["review_text"].str.casefold()
    df = df.drop_duplicates(subset=["_dedup"]).drop(columns=["_dedup"])
    df.to_csv(OUTPUT, index=False)
else:
    df = pd.DataFrame(columns=COLUMNS)

seen = set(df["review_text"].astype(str).str.strip().str.casefold())
print(f"Starting with {len(df)} unique reviews. Target: {TARGET}")

def generate_batch(count):
    prompt = f"""
Generate exactly {count} fictional customer reviews in a JSON object
with a "reviews" array. These are synthetic benchmark examples, not real feedback.

Each review object must contain:
- review_text: natural review, 5-80 words
- rating: integer 1 through 5
- sentiment: positive, negative, or neutral
- topic: one of {TOPICS}
- sarcasm: true or false
- negation: true or false
- language: English, Hinglish, or Other
- rating_text_mismatch: true or false
- prompt_injection: true or false

Include a balanced variety of ordinary reviews and difficult cases:
sarcasm, negation, mixed sentiment, Hinglish, informal spelling,
rating-text mismatches, and attempts to manipulate an AI system through
review text. Do not make every review an edge case.

Labels must match the actual text. A rating-text mismatch means the rating
conflicts with the sentiment expressed in the review text. Prompt injection
is true only if the review attempts to manipulate an AI system's instructions.
Use diverse products, scenarios, lengths, and wording. Avoid duplicates.
Do not copy real reviews or claim these are real customer feedback.

Return valid JSON only, with a "reviews" array and no Markdown.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate diverse synthetic benchmark data. Return valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=6000,
        reasoning_effort="low",
    )
    data = json.loads(response.choices[0].message.content)
    reviews = data.get("reviews")
    if not isinstance(reviews, list):
        raise ValueError("API response did not contain a reviews array")
    return reviews

def valid_row(review):
    if not isinstance(review, dict):
        return None

    text = str(review.get("review_text", "")).strip()
    rating = review.get("rating")

    if not text or isinstance(rating, bool) or rating not in [1, 2, 3, 4, 5]:
        return None
    if review.get("sentiment") not in ["positive", "negative", "neutral"]:
        return None
    if review.get("topic") not in TOPICS:
        return None
    if review.get("language") not in ["English", "Hinglish", "Other"]:
        return None
    if any(not isinstance(review.get(k), bool) for k in BOOLEAN_FIELDS):
        return None

    return {
        "review_text": text,
        "rating": rating,
        "sentiment": review["sentiment"],
        "topic": review["topic"],
        "sarcasm": review["sarcasm"],
        "negation": review["negation"],
        "language": review["language"],
        "rating_text_mismatch": review["rating_text_mismatch"],
        "prompt_injection": review["prompt_injection"],
        "source": "synthetic_amd_deepseek_v4_flash",
    }

while len(df) < TARGET:
    count = min(BATCH_SIZE, TARGET - len(df))
    added_rows = []
    last_error = None

    for attempt in range(6):
        try:
            reviews = generate_batch(count)
            batch_seen = set()

            for review in reviews:
                row = valid_row(review)
                if row is None:
                    continue

                key = row["review_text"].casefold()
                if key in seen or key in batch_seen:
                    continue

                added_rows.append(row)
                batch_seen.add(key)

            if added_rows:
                break

            last_error = "No valid, unique reviews returned in this batch."

        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"

        wait = min(60, 5 * (2 ** attempt)) + random.uniform(0, 2)
        print(f"Attempt {attempt + 1}/6 failed: {last_error}")
        if attempt < 5:
            print(f"Retrying in {wait:.1f} seconds...")
            time.sleep(wait)

    if not added_rows:
        print("Stopped safely; rerun later to resume.")
        print(f"Progress saved: {len(df)}/{TARGET}")
        break

    df = pd.concat(
        [df, pd.DataFrame(added_rows, columns=COLUMNS)],
        ignore_index=True,
    )
    df["_dedup"] = df["review_text"].astype(str).str.casefold()
    df = df.drop_duplicates(subset=["_dedup"]).drop(columns=["_dedup"])
    df.to_csv(OUTPUT, index=False)

    seen = set(df["review_text"].astype(str).str.casefold())
    print(f"Saved {len(df)}/{TARGET} unique reviews to {OUTPUT}")

    # Approximately 13-17 requests per minute, below the documented limit.
    time.sleep(3.5 + random.uniform(0, 1))

if len(df) >= TARGET:
    print(f"Finished: {len(df)} unique reviews saved to {OUTPUT}")
else:
    print(f"Current progress: {len(df)}/{TARGET}. Rerun to resume.")
