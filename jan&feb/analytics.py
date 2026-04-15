"""analytics.py — Interaction Logging & Analytics Dashboard."""

import json, os, csv
from datetime import datetime

LOG_FILE = "sit_chat_logs.json"
CSV_FILE = "sit_analytics.csv"


def log_interaction(query, intent, confidence, method, answer_found):
    entry = {
        "timestamp":    datetime.now().isoformat(),
        "query":        query,
        "intent":       intent,
        "confidence":   round(float(confidence), 4),
        "method":       method,
        "answer_found": bool(answer_found),
    }
    logs = _load()
    logs.append(entry)
    with open(LOG_FILE, 'w') as f:
        json.dump(logs, f, indent=2)


def _load():
    if not os.path.exists(LOG_FILE):
        return []
    with open(LOG_FILE) as f:
        try:
            return json.load(f)
        except Exception:
            return []


def analyze_logs() -> str:
    logs = _load()
    if not logs:
        return "No interactions logged yet. Start chatting!"

    total  = len(logs)
    found  = sum(1 for l in logs if l.get('answer_found'))
    avg_c  = sum(l.get('confidence', 0) for l in logs) / total

    intent_counts: dict = {}
    method_counts: dict = {}
    for l in logs:
        k = l.get('intent', 'unknown')
        intent_counts[k] = intent_counts.get(k, 0) + 1
        m = l.get('method', 'unknown')
        method_counts[m] = method_counts.get(m, 0) + 1

    intent_lines = '\n'.join(
        f"  {k:<18} {v:>3} ({v/total*100:.0f}%)"
        for k, v in sorted(intent_counts.items(), key=lambda x: -x[1])
    )
    method_lines = '\n'.join(
        f"  {k:<12} {v:>3}"
        for k, v in sorted(method_counts.items(), key=lambda x: -x[1])
    )
    recent = '\n'.join(
        f"  [{l['timestamp'][:19]}] [{l.get('intent','?'):<12}] {l['query']}"
        for l in logs[-5:]
    )

    return f"""
╔══════════════════════════════════════╗
║    SIT NAGPUR CHATBOT ANALYTICS      ║
╚══════════════════════════════════════╝
  Total Interactions : {total}
  Answered           : {found}  ({found/total*100:.1f}%)
  Unanswered         : {total-found}
  Avg Confidence     : {avg_c:.3f}

  Intent Distribution:
{intent_lines}

  Retrieval Methods:
{method_lines}

  Recent 5 Queries:
{recent}
""".strip()


def export_csv():
    logs = _load()
    if not logs:
        return "No data to export."
    keys = list(logs[0].keys())
    with open(CSV_FILE, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(logs)
    return f"✅ Exported {len(logs)} records → {CSV_FILE}"


def get_log_count():
    return len(_load())
