import argparse
import csv
import math
from collections import defaultdict

from lib.utils_task1 import load_pairs_csv
from lib.eval_task1 import macro_eval

def load_scores(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "table_id": r["table_id"],
                "text_id": r["text_id"],
                "score": float(r["score"]),
            })
    return rows


def zscore_by_table(score_rows):
    by_table = defaultdict(list)
    for r in score_rows:
        by_table[r["table_id"]].append(r)

    out = {}
    for table_id, rows in by_table.items():
        vals = [r["score"] for r in rows]
        mean = sum(vals) / len(vals)
        var = sum((x - mean) ** 2 for x in vals) / len(vals)
        std = math.sqrt(var) if var > 1e-12 else 1.0

        for r in rows:
            out[(r["table_id"], r["text_id"])] = (r["score"] - mean) / std

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--score1", required=True)
    ap.add_argument("--score2", required=True)
    ap.add_argument("--pairs_path", required=True)
    ap.add_argument("--alpha", type=float, default=0.2)
    ap.add_argument("--k_eval", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--test", action="store_true", help="Whether to run in test mode without scoring")
    ap.add_argument("--save_rankings", default=None, help="Optional path to save rankings as CSV: table_id,text_id,score,rank")
    args = ap.parse_args()

    score1_rows = load_scores(args.score1)
    score2_rows = load_scores(args.score2)
    pairs = load_pairs_csv(args.pairs_path)

    score1_z = zscore_by_table(score1_rows)
    score2_z = zscore_by_table(score2_rows)

    cand_by_table = defaultdict(list)
    rel_by_table = defaultdict(set)


    for text_id, table_id, label in pairs:
        cand_by_table[table_id].append(text_id)
        if label == 1:
            rel_by_table[table_id].add(text_id)

    all_rankings = {}
    saved_rows = []

    for table_id, cand_text_ids in cand_by_table.items():
        scored = []
        for text_id in cand_text_ids:
            s1 = score1_z.get((table_id, text_id), 0.0)
            s2 = score2_z.get((table_id, text_id), 0.0)
            final_score = s1 + args.alpha * s2
            scored.append((text_id, final_score))

        ranked_items = sorted(scored, key=lambda x: x[1], reverse=True)
        ranked_text_ids = [text_id for text_id, _ in ranked_items]
        all_rankings[table_id] = ranked_text_ids

        if args.save_rankings:
            for rank, (text_id, score) in enumerate(ranked_items, start=1):
                saved_rows.append((table_id, text_id, score, rank))

    if not args.test:
        report = macro_eval(all_rankings, rel_by_table, args.k_eval)

        print(f"=== Fused ranking metrics (alpha={args.alpha}) ===")
        for k in args.k_eval:
            m = report[k]
            print(
                f"k={k:>3} | P@k={m['P@k']:.4f} | "
                f"R@k={m['R@k']:.4f} | "
                f"F1@k={m['F1@k']:.4f} | "
                f"MRR@k={m['MRR@k']:.4f} | "
                f"#tables={m['num_tables']}"
            )

    if args.save_rankings:
        with open(args.save_rankings, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["table_id", "text_id", "score", "rank"])
            w.writerows(saved_rows)
        print(f"Saved rankings to: {args.save_rankings}")


if __name__ == "__main__":
    main()
