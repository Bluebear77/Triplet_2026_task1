# task1_bm25_baseline.py
# Usage:
#   python task1_bm25_baseline.py \
#     --tables_path tables.jsonl \
#     --texts_path texts.jsonl \
#     --pairs_path pairs.csv \
#     --k_rows 15 \
#     --k_eval 5 10 20

import pandas as pd
import argparse
import json
from collections import Counter, defaultdict
from lib.bm25 import build_bm25, bm25_score
from lib.utils import normalize_text, extract_relevant_subtable, flatten_df
from lib.utils_task1 import load_json_or_jsonl, load_pairs_csv, table_to_df, table_to_text, table_to_text_colfilt, annotation_penalty
from lib.eval_task1 import macro_eval


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--output_jsonl", required=True, help="Path to save rankings as JSONL: table_id,text_id,score")
    args = ap.parse_args()

    rows = load_json_or_jsonl(args.data_jsonl)
    saved_rows = []

    for r in rows:        
        q = r["title"] + ". " + r["table"]

        index = build_bm25({r["text_id"]: r["text"]})

        scores = bm25_score(index, q)

        saved_rows.append({"table_id": r["table_id"], "text_id": r["text_id"], "score": scores[r["text_id"]]})


    # Save rankings
    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in saved_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved rankings to: {args.output_jsonl}")

if __name__ == "__main__":
    main()
