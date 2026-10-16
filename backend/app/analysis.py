"""Review analysis, trend detection, alert scoring, momentum. Pure Python, no dependencies.
Everything here is rule-based and explainable; it is NOT a validated predictor.
"""
import re
import math
import random
import datetime as dt

TOPICS = {
    "quality": ["quality", "material", "build", "finish", "solid", "cheap", "durable", "sturdy", "broke", "crack"],
    "cooling": ["cool", "cooling", "airflow", "breeze", "chilling", "fan", "heat", "hot"],
    "price": ["price", "value", "money", "worth", "expensive", "cost", "cheap", "overpric"],
    "delivery": ["deliver", "shipping", "courier", "arrived", "fast", "late", "delay"],
    "sound": ["sound", "bass", "audio", "volume", "treble", "noise", "loud", "music"],
    "battery": ["battery", "charg", "backup", "drain", "power"],
    "packaging": ["packag", "box", "seal", "damage"],
    "support": ["support", "service", "warranty", "replacement", "repair"],
    "comfort/fit": ["comfort", "uncomfort", "fit", "tight", "size", "pain"]
}

WEIGHT = {
    "quality": 3, "cooling": 3, "battery": 3, "durability": 3,
    "sound": 2, "delivery": 2, "support": 2, "comfort/fit": 2,
    "price": 1, "packaging": 1
}

POS = [
    "great", "good", "excellent", "love", "amazing", "perfect", "comfort", "fast",
    "crisp", "clear", "solid", "happy", "helpful", "quick", "nice", "best", "neat",
    "worth", "durable", "lasts", "super", "fine", "chilling", "effective", "awesome"
]
NEG = [
    "drain", "dies", "poor", "bad", "broke", "crack", "late", "delay", "damage",
    "slow", "worst", "terrible", "disappoint", "uncomfort", "tight", "pain",
    "useless", "weak", "muffled", "overpric", "stopped", "fail", "useless", "waste"
]
NEGATORS = {"not", "no", "never", "isn't", "wasn't", "doesn't", "don't", "didn't"}
NEXT = {
    "battery": "Check recent production batches, supplier/cell changes and warranty claims.",
    "quality": "Inspect returns and factory defect reports; check supplier molding/materials.",
    "cooling": "Evaluate fan motor specifications and thermal ambient tolerances.",
    "delivery": "Compare courier performance per marketplace and investigate transit delays.",
    "sound": "Audit audio driver quality assurance testing and firmware tuning.",
    "price": "Review competitive market pricing against alternative listings."
}
ALT = "Alternative explanations: promotional customer influx, changed review prompts, seasonal spikes, or low sample size."


def _tok(s):
    return re.findall(r"[a-z']+", s.lower())


def _sent(tokens):
    net = hits = 0
    for i, t in enumerate(tokens):
        s = 1 if any(t.startswith(p) for p in POS) and not t.startswith("uncomfort") and not t.endswith("ly") else 0
        if any(t.startswith(n) for n in NEG):
            s = -1
        if s:
            if i and tokens[i - 1] in NEGATORS:
                s = -s
            net += s
            hits += 1
    return net, hits


def analyze(text, rating=None):
    topics, tot, hits_all = [], 0, 0
    for part in re.split(r"[.!?;]|\bbut\b|\bhowever\b", text.lower()):
        tk = _tok(part)
        found = [t for t, kws in TOPICS.items() if any(w.startswith(k) for w in tk for k in kws)]
        net, hits = _sent(tk)
        tot += net
        hits_all += hits
        for t in found:
            if net and not any(x["topic"] == t and x["sent"] == ("pos" if net > 0 else "neg") for x in topics):
                topics.append({"topic": t, "sent": "pos" if net > 0 else "neg"})
    kinds = {x["sent"] for x in topics}
    if kinds == {"pos", "neg"}:
        sent = "neutral"
    elif tot > 0:
        sent = "positive"
    elif tot < 0:
        sent = "negative"
    elif rating is not None and hits_all == 0:
        sent = "positive" if rating >= 4 else "negative" if rating <= 2 else "neutral"
    else:
        sent = "neutral"
    conf = round(min(0.9, 0.35 + 0.15 * hits_all), 2)
    mismatch = int(rating is not None and ((rating >= 4 and "neg" in kinds) or (rating <= 2 and sent == "positive")))
    return {"sentiment": sent, "conf": conf, "topics": topics, "mismatch": mismatch}


def _z(x1, n1, x2, n2):
    p = (x1 + x2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return (x1 / n1 - x2 / n2) / se if se else 0.0


def detect(rows, today, W=7, B=28, min_recent=15, min_base=30, min_hits=4, min_diff=0.08, min_z=2.0):
    """Compare last W days with the preceding B days: share of a product's reviews mentioning a topic

    with a given sentiment. Safely handles datasets without timestamps.
    """
    dated_rows = [r for r in rows if r.get("date") and isinstance(r["date"], (dt.date, dt.datetime))]
    if not dated_rows:
        return [], [
            "Historical trend detection requires dated records. The imported Flipkart and Amazon Kaggle datasets "
            "do not contain review timestamps. Switch to 'Synthetic Trend Demo' mode to preview the anomaly detection engine."
        ]

    alerts, notes = [], []
    r0 = today - dt.timedelta(days=W)
    b0 = r0 - dt.timedelta(days=B)
    for prod in sorted({r["product"] for r in dated_rows if r.get("product")}):
        pr = [r for r in dated_rows if r["product"] == prod]
        rec = [r for r in pr if r["date"] > r0]
        base = [r for r in pr if b0 < r["date"] <= r0]
        n1, n2 = len(rec), len(base)
        if n1 < min_recent or n2 < min_base:
            notes.append(f"{prod}: insufficient data for trend detection ({n1} recent / {n2} baseline reviews; need {min_recent}/{min_base}).")
            continue
        for t in TOPICS:
            for kind in ("neg", "pos"):
                hit = lambda r: any(x["topic"] == t and x["sent"] == kind for x in r["topics"])
                x1, x2 = sum(map(hit, rec)), sum(map(hit, base))
                p1, p2 = x1 / n1, x2 / n2
                d = p1 - p2
                z = _z(x1, n1, x2, n2)
                if x1 < min_hits or d < min_diff or z < min_z:
                    continue
                mid = r0 + dt.timedelta(days=4)
                h1 = [r for r in rec if r["date"] <= mid]
                h2 = [r for r in rec if r["date"] > mid]
                sustained = len(h1) >= 3 and len(h2) >= 3 and all(sum(map(hit, h)) / len(h) > p2 for h in (h1, h2))
                sev = (WEIGHT.get(t, 2) / 3) * min(d / 0.2, 1) if kind == "neg" else 0.3
                conf = min(z / 4, 1) * min(n1 / 30, 1)
                score = 100 * (.3 * sev + .25 * min(d / .3, 1) + .2 * conf + .15 * (1 if sustained else .5) + .1 * min(p1 / .3, 1))
                plats = {}
                for r in rec:
                    if hit(r):
                        plats[r["platform"]] = plats.get(r["platform"], 0) + 1
                ev = sorted([r for r in rec if hit(r)], key=lambda r: r["date"], reverse=True)
                seen, ex = set(), []
                for r in ev:
                    if r["platform"] not in seen or len(ex) < 3:
                        ex.append(r)
                        seen.add(r["platform"])
                    if len(ex) == 3:
                        break
                lab = "complaints" if kind == "neg" else "praise"
                alerts.append({
                    "key": f"{prod}|{t}|{kind}",
                    "product": prod,
                    "topic": t,
                    "kind": kind,
                    "title": f"{t.capitalize()} {lab} {'rising' if kind == 'neg' else 'growing'} for {prod}",
                    "recent": f"{r0 + dt.timedelta(days=1)} to {today}",
                    "baseline": f"{b0 + dt.timedelta(days=1)} to {r0}",
                    "x1": x1, "n1": n1, "p1": round(p1, 3), "x2": x2, "n2": n2, "p2": round(p2, 3), "z": round(z, 2),
                    "persistence": "Sustained across recent window" if sustained else "Temporary / concentrated spike",
                    "severity": ("High" if sev >= .6 else "Medium" if sev >= .3 else "Low") if kind == "neg" else "Opportunity",
                    "confidence": "High" if conf >= .6 else "Medium" if conf >= .35 else "Low: verify",
                    "score": round(score, 1),
                    "platforms": plats,
                    "evidence": [{"date": str(r["date"]), "platform": r["platform"], "rating": r.get("rating"), "text": r.get("text", "")} for r in ex],
                    "next": NEXT.get(t, "Read the supporting reviews; check whether the issue is tied to a batch, listing or marketplace."),
                    "alt": ALT,
                    "disclaimer": "This is a statistical signal that needs human investigation, not a confirmed defect."
                })
    alerts.sort(key=lambda a: -a["score"])
    return alerts, notes


def momentum(rows, today):
    dated_rows = [r for r in rows if r.get("date") and isinstance(r["date"], (dt.date, dt.datetime))]
    if not dated_rows:
        return [{
            "product": "Current Marketplace Catalog",
            "state": "Requires Dated Records",
            "indicators": [],
            "n": [0, 0]
        }]

    out = []
    for prod in sorted({r["product"] for r in dated_rows if r.get("product")}):
        pr = [r for r in dated_rows if r["product"] == prod]
        cur = [r for r in pr if r["date"] > today - dt.timedelta(days=28)]
        prev = [r for r in pr if today - dt.timedelta(days=56) < r["date"] <= today - dt.timedelta(days=28)]
        if len(cur) < 20 or len(prev) < 20:
            out.append({"product": prod, "state": "Insufficient data", "indicators": [], "n": [len(cur), len(prev)]})
            continue
        avg = lambda L: sum(r["rating"] for r in L if r.get("rating") is not None) / max(1, sum(1 for r in L if r.get("rating") is not None))
        share = lambda L, s: sum(r["sentiment"] == s for r in L) / max(1, len(L))
        ind = []
        def add(name, a, b, bad_if_down, thr, fmt):
            d = a - b
            sig = "flat" if abs(d) < thr else ("worse" if (d < 0) == bad_if_down else "better")
            ind.append({"name": name, "now": fmt(a), "before": fmt(b), "signal": sig})
        add("Average rating", avg(cur), avg(prev), True, .25, lambda v: f"{v:.2f}")
        add("Negative review share", share(cur, "negative"), share(prev, "negative"), False, .08, lambda v: f"{v:.0%}")
        add("Positive review share", share(cur, "positive"), share(prev, "positive"), True, .08, lambda v: f"{v:.0%}")
        add("Review volume (context only)", len(cur), len(prev), False, 10, str)
        w = sum(i["signal"] == "worse" for i in ind)
        b = sum(i["signal"] == "better" for i in ind)
        state = "Likely declining" if w >= 2 else "Emerging decline" if w == 1 else "Gaining momentum" if b >= 2 else "Stable"
        out.append({"product": prod, "state": state, "indicators": ind, "n": [len(cur), len(prev)]})
    return out


# ---------- synthetic demo data (FICTIONAL / CLEARLY LABELED) ----------
T = {
    "battery": (
        ["Battery lasts a full day, great backup.", "Charging is fast and the battery is excellent."],
        ["Battery drains too quickly, poor backup.", "Battery dies within two hours, terrible.", "Battery stopped holding charge, disappointed."]
    ),
    "cooling": (
        ["Cooling is fast and chilled airflow is great.", "Keeps the entire room cool comfortably."],
        ["Cooling stopped working, water pump failure.", "Airflow is hot and weak, very poor cooling."]
    ),
    "sound": (
        ["Sound quality is excellent with crisp, clear bass.", "Love the audio, perfect volume."],
        ["Sound is muffled and the bass is weak.", "Audio quality is poor, very disappointing."]
    ),
    "delivery": (
        ["Delivery was quick and on time.", "Arrived early, nice delivery."],
        ["Delivery was late and courier damaged the packaging.", "Shipping delayed by a week, terrible."]
    ),
    "packaging": (
        ["Packaging was neat and the box looked great."],
        ["Box arrived damaged, packaging was torn."]
    ),
    "quality": (
        ["Build quality is solid and durable."],
        ["Plastic cracked and knob broke after three weeks."]
    ),
    "comfort/fit": (
        ["Very comfortable, fits perfectly for long hours."],
        ["Uncomfortable after an hour, too tight and causes pain."]
    ),
    "support": (
        ["Customer support was helpful and quick."],
        ["Support was useless, no replacement offered."]
    ),
    "price": (
        ["Great value for the price."],
        ["Overpriced and not worth the money."]
    )
}


def demo_rows(today, seed=7, inject=True, decline=True):
    """Generates synthetic review records clearly labeled for trend demo."""
    rnd = random.Random(seed)
    out = []
    plan = {
        "AuraBeat Wireless Headphones": {"Amazon": 3, "Flipkart": 2, "Own Website": 1},
        "AuraFit Earbuds": {"Amazon": 2, "Flipkart": 1.5}
    }
    names, wts = list(T), [3 if t in ("battery", "sound") else 1 for t in T]
    for prod, pl in plan.items():
        for plat, rate in pl.items():
            for ago in range(119, -1, -1):
                for _ in range(sum(rnd.random() < rate / 3 for _ in range(3))):
                    ts = []
                    want = rnd.choice([1, 2, 2, 3])
                    while len(ts) < want:
                        c = rnd.choices(names, wts)[0]
                        if c not in ts:
                            ts.append(c)
                    parts, negs = [], 0
                    for t in ts:
                        pn = .06
                        if inject and prod.startswith("AuraBeat") and t == "battery" and ago < 7:
                            pn = .5 if plat != "Own Website" else .12
                        if decline and prod.startswith("AuraFit") and t in ("durability", "comfort/fit", "sound", "battery"):
                            pn = .06 + .5 * max(0, 56 - ago) / 56
                        neg = rnd.random() < pn
                        negs += neg
                        parts.append(rnd.choice(T[t][1 if neg else 0]))
                    rating = 5 if rnd.random() < .6 else 4
                    if negs and negs == len(ts):
                        rating = rnd.choice([1, 2])
                    elif negs:
                        rating = rnd.choice([3, 4])
                    if rnd.random() < .02:
                        rating = 5 if negs else 1
                    out.append({
                        "platform": plat,
                        "product": prod,
                        "date": (today - dt.timedelta(days=ago)).isoformat(),
                        "rating": rating,
                        "text": " ".join(parts),
                        "is_synthetic": 1,
                        "source": "demo"
                    })
    return out
