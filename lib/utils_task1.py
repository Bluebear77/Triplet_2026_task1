import re
import json
import csv
import random
import pandas as pd
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
from .utils import tokenize, normalize_text

# -------------------------
# I/O helpers
# -------------------------
def load_json_or_jsonl(path: str) -> List[dict]:
    """Load either a JSON array file or a JSONL file."""
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        # jsonl
        out = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out


def load_pairs_csv(path: str) -> List[Tuple[str, str, int]]:
    """Return list of (text_id, table_id, label)."""
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Accept header order variations
        # Expected: text_id,table_id,label
        idx = {name: i for i, name in enumerate(header)}
        for row in reader:
            text_id = row[idx["text_id"]]
            table_id = row[idx["table_id"]]
           # if "label" not in idx:
           #     label = None
           #  else:
           #     label = int(row[idx["label"]])
            if "label" not in idx or row[idx["label"]].strip() == "":
                label = None
            else:
                label = int(row[idx["label"]])
            pairs.append((text_id, table_id, label))
    return pairs

def split_by_table(rows, valid_ratio=0.1, seed=42):
    rng = random.Random(seed)
    table_ids = sorted({row["table_id"] for row in rows})
    rng.shuffle(table_ids)

    n_valid = max(1, int(len(table_ids) * valid_ratio))
    valid_tables = set(table_ids[:n_valid])

    train_rows = []
    valid_rows = []

    for row in rows:
        if row["table_id"] in valid_tables:
            valid_rows.append(row)
        else:
            train_rows.append(row)

    return train_rows, valid_rows

_CELL_CLEAN_RE = re.compile(r"\s+")
_NUMISH_RE = re.compile(r"^[\W_]*\d+(?:[\W_]+\d+)*[\W_]*$")  # numbers with punctuation, e.g., "4 - 2", "30 - 11 - 10"
_TOO_SHORT_ALPHA_RE = re.compile(r"^[a-zA-Z]{1,2}$")

def clean_cell(s: str) -> str:
    s = str(s)
    s = s.replace("\n", " ").strip()
    s = _CELL_CLEAN_RE.sub(" ", s)
    return s

def is_numish(s: str) -> bool:
    s2 = s.strip()
    if not s2:
        return True
    # Mostly numbers/symbols
    return bool(_NUMISH_RE.match(s2))

def keep_cell(s: str) -> bool:
    s = clean_cell(s)
    if not s:
        return False
    if is_numish(s):
        return False
    # Drop extremely short alpha tokens (often noise)
    if _TOO_SHORT_ALPHA_RE.match(s):
        return False
    # If it's all punctuation, drop
    if not re.search(r"[A-Za-z]", s):
        return False
    return True

def table_to_text(t: dict, k_rows: int = 15, max_vals_per_col: int = 8, max_context: int = 2) -> str:
    """
    Make a 'dense' query text for BM25:
      - always include title + headers
      - from rows: keep only informative string cells (drop numeric-ish)
      - cap unique values per column to avoid noise
    """
    title = clean_cell(t.get("title", ""))
    header = t.get("header", []) or []
    rows = t.get("rows", []) or []

    ctx = t.get("context", [])

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    
    if max_context > 0 and ctx:
        for c in ctx[:max_context]:
            q = c.get("question", "")
            a = c.get("answer", "")

            if q:
                parts.append(q)

            if a:
                parts.append(a)

        return normalize_text(" ".join(parts))
    
    # fallback
    if header:
        parts.append("HEADER: " + " | ".join(clean_cell(h) for h in header))

    # Collect per-column unique informative values
    col_vals = defaultdict(list)       # col -> [vals in order]
    col_seen = defaultdict(set)

    for r in rows[:k_rows]:
        for j, cell in enumerate(r):
            c = clean_cell(cell)
            if not keep_cell(c):
                continue
            if c in col_seen[j]:
                continue
            col_seen[j].add(c)
            col_vals[j].append(c)

    # Emit values per column with cap
    if header:
        for j, h in enumerate(header):
            vals = col_vals.get(j, [])[:max_vals_per_col]
            if not vals:
                continue
            parts.append(f"COL {clean_cell(h)}: " + "; ".join(vals))
    else:
        # if no header, just dump values
        for j, vals in col_vals.items():
            vals = vals[:max_vals_per_col]
            if vals:
                parts.append(f"COL{j}: " + "; ".join(vals))

    return normalize_text("\n".join(parts))


def table_to_df(t: dict) -> dict:
    """
    Make a 'dense' query text for BM25:
      - always include title + headers
      - from rows: keep only informative string cells (drop numeric-ish)
      - cap unique values per column to avoid noise
    """
    title = clean_cell(t["title"])
    header = t["header"]

    data = {
        "id": t["id"],
        "title": title,
        "ctx": t["context"],
    }
    
    norm_rows = []
    for r in t["rows"]:
        row = []
        for j, cell in enumerate(r):
            row.append(normalize_text(cell))
        norm_rows.append(row)

    data["table"] = pd.DataFrame(norm_rows, columns=header, index=None).dropna(axis=1, how='all')

    return data


NOISY_COL_PAT = re.compile(r"(note|notes|remark|remarks|comment|comments|description|desc|footnote|details)", re.I)

def table_to_text_colfilt(t: dict, k_rows: int = 50, max_vals_per_col: int = 12) -> str:
    title = clean_cell(t.get("title", ""))
    header = t.get("header", []) or []
    rows = t.get("rows", []) or []

    parts = []
    if title:
        parts.append(f"TITLE: {title}")

    # Decide which columns to keep (drop notes-like columns)
    keep_cols = []
    for j, h in enumerate(header):
        h_clean = clean_cell(h)
        if NOISY_COL_PAT.search(h_clean):
            continue
        keep_cols.append(j)

    if header:
        parts.append("HEADER: " + " | ".join(clean_cell(header[j]) for j in keep_cols) if keep_cols else "HEADER: " + " | ".join(clean_cell(x) for x in header))

    # Collect values from kept columns
    col_vals = defaultdict(list)
    col_seen = defaultdict(set)

    # rows cap is less important; many tables have <20 rows anyway
    for r in rows[:k_rows]:
        for j in keep_cols if keep_cols else range(len(r)):
            if j >= len(r):
                continue
            c = clean_cell(r[j])
            if not keep_cell(c):  # uses numeric-ish / non-alpha filter
                continue
            if c in col_seen[j]:
                continue
            col_seen[j].add(c)
            col_vals[j].append(c)

    # Emit per column
    if header:
        for j in keep_cols if keep_cols else range(len(header)):
            vals = col_vals.get(j, [])[:max_vals_per_col]
            if not vals:
                continue
            parts.append(f"COL {clean_cell(header[j])}: " + "; ".join(vals))
    else:
        for j, vals in col_vals.items():
            vals = vals[:max_vals_per_col]
            if vals:
                parts.append(f"COL{j}: " + "; ".join(vals))

    return normalize_text("\n".join(parts))


def table_to_text_simple(t: dict, k_rows: int = 15) -> str:
    title = t.get("title", "")
    header = t.get("header", []) or []
    rows = t.get("rows", []) or []

    parts = []
    if title:
        parts.append(f"TITLE: {title}")
    if header:
        parts.append("HEADER: " + " | ".join(map(str, header)))

    # take first k_rows
    for r in rows[:k_rows]:
        parts.append("ROW: " + " | ".join(map(str, r)))

    # NOTE: we intentionally ignore "context" here until rules are confirmed
    return normalize_text("\n".join(parts))

def annotation_penalty(text: str) -> float:
    """
    Return multiplicative penalty in (0, 1].
    Heuristics tuned for Wikipedia-style footnotes/captions.
    """
    t = normalize_text(text).lower()
    toks = tokenize(t)
    n = len(toks)

    pen = 1.0

    # Very short paragraphs are often notes/captions
    if n <= 6:
        pen *= 0.55
    elif n <= 10:
        pen *= 0.75

    # Footnote / note markers
    if "†" in text or "‡" in text:
        pen *= 0.80

    # Typical note-like vocabulary
    note_terms = [
        "denotes", "reflect", "only", "stats", "spent time", "traded", "mid-season",
        "before joining", "before", "after", "career totals", "note:", "notes:"
    ]
    if any(term in t for term in note_terms):
        pen *= 0.80

    return pen
