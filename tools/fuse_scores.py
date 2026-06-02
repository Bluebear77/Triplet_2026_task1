import argparse
import json
import math
from collections import defaultdict

from lib.utils_task1 import load_pairs_csv
from lib.eval_task1 import macro_eval

def load_scores(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
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
    ap.add_argument("--score1", required=True, help="Path to first score file (jsonl with table_id,text_id,score)")
    ap.add_argument("--score2", required=True, help="Path to second score file (jsonl with table_id,text_id,score)")
    ap.add_argument("--pairs_path", required=True, help="Path to pairs file (csv with text_id,table_id,label)")
    ap.add_argument("--alpha", type=float, default=1.0, help="Weight for the second score in the fusion")
    ap.add_argument("--zscore", action="store_true", help="Whether to z-score the scores by table before fusion")
    ap.add_argument("--save_scores_jsonl", required=True, help="Path to save scores as jsonl: table_id,text_id,score")
    args = ap.parse_args()

    score1_rows = load_scores(args.score1)
    score2_rows = load_scores(args.score2)
    pairs = load_pairs_csv(args.pairs_path)

    if args.zscore:
        score1_z = zscore_by_table(score1_rows)
        score2_z = zscore_by_table(score2_rows)
    else:
        score1_z = {(r["table_id"], r["text_id"]): r["score"] for r in score1_rows}
        score2_z = {(r["table_id"], r["text_id"]): r["score"] for r in score2_rows}

    cand_by_table = defaultdict(list)


    for text_id, table_id, __ in pairs:
        cand_by_table[table_id].append(text_id)

   
    all_scores = []
    for table_id, cand_text_ids in cand_by_table.items():
        for text_id in cand_text_ids:
            all_scores.append({
                "table_id": table_id,
                "text_id": text_id,
                "score": (score1_z[(table_id, text_id)] + args.alpha * score2_z[(table_id, text_id)]) / (1 + args.alpha)
            })

    with open(args.save_scores_jsonl, "w", encoding="utf-8") as f:
        for r in all_scores:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved scores to: {args.save_scores_jsonl}")


if __name__ == "__main__":
    main()
