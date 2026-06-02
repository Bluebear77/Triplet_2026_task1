import argparse
from collections import defaultdict
import csv

from task1_bm25_baseline import (
    load_pairs_csv,
    macro_eval,
)

def convert_from_saved_rankings_to_all_rankings(saved_rankings_path):
    all_rankings = defaultdict(list)  # table_id -> [text_id...]
    all_scores = defaultdict(list)    # table_id -> [score...]
    with open(saved_rankings_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table_id = row["table_id"]
            text_id = row["text_id"]
                
            all_scores[table_id].append(float(row["score"]))
            all_rankings[table_id].append(text_id)
        
    # スコアの降順で並び替え
    for table_id in all_rankings:
        all_rankings[table_id] = [x for _, x in sorted(zip(all_scores[table_id], all_rankings[table_id]), reverse=True)]
    return all_rankings


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_path", required=True)
    ap.add_argument("--k_eval", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--saved_rankings", required=True, help="Path to saved rankings as CSV: table_id,text_id,score,rank")
    args = ap.parse_args()

    pairs = load_pairs_csv(args.pairs_path)

    # Build candidate sets from pairs (safe assumption for training: evaluate within provided candidate pool)
    cand_by_table = defaultdict(list)   # table_id -> [text_id...]
    rel_by_table = defaultdict(set)     # table_id -> {relevant text_ids}
    for text_id, table_id, label in pairs:
        cand_by_table[table_id].append(text_id)
        if label == 1:
            rel_by_table[table_id].add(text_id)

    # For each table, BM25 rank candidates (build per-table index over its candidate paragraphs)
    all_rankings = convert_from_saved_rankings_to_all_rankings(args.saved_rankings)

    # Evaluate
    report = macro_eval(all_rankings, rel_by_table, args.k_eval)
    print("=== Macro-averaged metrics over tables (candidate pool from pairs.csv) ===")
    for k in args.k_eval:
        m = report[k]
        print(f"k={k:>3} | P@k={m['P@k']:.4f} | R@k={m['R@k']:.4f} | F1@k={m['F1@k']:.4f} | #tables={m['num_tables']}")

if __name__ == "__main__":
    main()
