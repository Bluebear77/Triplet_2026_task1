import pandas as pd
import argparse
import json
from collections import Counter, defaultdict
from lib.utils import normalize_text, extract_relevant_subtable, flatten_df
from lib.utils_task1 import load_json_or_jsonl, load_pairs_csv, table_to_df

# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables_path", type=str, required=True)
    ap.add_argument("--texts_path", type=str, required=True)
    ap.add_argument("--pairs_path", type=str, required=True)
    ap.add_argument("--k_rows", type=int, default=3)
    ap.add_argument("--table_input_type", type=str, default="structured", choices=["text", "structured", "keyValue"])
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--output_jsonl", type=str, required=True)
    args = ap.parse_args()

    tables = load_json_or_jsonl(args.tables_path)
    texts = load_json_or_jsonl(args.texts_path)
    pairs = load_pairs_csv(args.pairs_path)

    tables_by_id = {t["id"]: t for t in tables}
    texts_by_id = {x["text_id"]: x for x in texts}

    # Build candidate sets from pairs (safe assumption for training: evaluate within provided candidate pool)
    cand_by_table = defaultdict(list)   # table_id -> [text_id...]
    if args.test:
        for text_id, table_id, __ in pairs:
            cand_by_table[table_id].append(text_id)
    else:
        rel_by_table = defaultdict(set)     # table_id -> {relevant text_ids}
        for text_id, table_id, label in pairs:
            cand_by_table[table_id].append(text_id)
            if label == 1:
                rel_by_table[table_id].add(text_id)

    # Precompute serialized table texts
    table_query = {}
    for table_id in cand_by_table.keys():
        t = tables_by_id[table_id]
        if t is None:
            continue
        table_query[table_id] = table_to_df(t)

    out_rows = []

    for table_id, cand_text_ids in cand_by_table.items():
        df = table_query[table_id]["table"]
        # docs and extracted table
        result_dfs = []
        result_dfs_all = []
        docs = {}
        for tid in cand_text_ids:
            tx = texts_by_id[tid]
            if tx is None:
                continue
            docs[tid] = normalize_text(tx["text"])

            df_ex, score = extract_relevant_subtable(df, docs[tid], "notUsedInThisCase", max_rows=args.k_rows)
            df_ex = df_ex.copy()
            if score > 0:
                result_dfs.append(df_ex)
            result_dfs_all.append(df_ex)

        merged_df = pd.concat(result_dfs if result_dfs != [] else result_dfs_all).drop_duplicates()

        for tid in cand_text_ids:
            row = {
                "table_id": table_id,
                "text_id": tid,
                "title": table_query[table_id]["title"],
                "text": docs[tid],
                "table": flatten_df(merged_df, table_input_type=args.table_input_type),
                }
            if not args.test:
                row["relevant_text_ids"] = list(rel_by_table[table_id])
                row["label"] = 1 if tid in rel_by_table[table_id] else 0
            out_rows.append(row)

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
