import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from lib.eval_task1 import macro_eval


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at {path}:{lineno}: {e}") from e

            for key in ["table_id", "text_id", "label", "score"]:
                if key not in row:
                    raise KeyError(f"Missing key '{key}' at {path}:{lineno}")
            rows.append(row)
    return rows


def build_rankings_and_gold(rows):
    by_table = defaultdict(list)
    rel_by_table = defaultdict(set)

    for row in rows:
        table_id = str(row["table_id"])
        text_id = str(row["text_id"])
        label = int(row["label"])
        score = float(row["score"])

        by_table[table_id].append((score, text_id))
        if label == 1:
            rel_by_table[table_id].add(text_id)

    all_rankings = {}
    for table_id, scored_texts in by_table.items():
        # Sort by score descending. text_id is used as a deterministic tie-breaker.
        ranked = [text_id for score, text_id in sorted(scored_texts, key=lambda x: (-x[0], x[1]))]
        all_rankings[table_id] = ranked

    return all_rankings, rel_by_table


def classification_metrics(rows, threshold):
    labels = np.array([int(row["label"]) for row in rows])
    scores = np.array([float(row["score"]) for row in rows])
    preds = (scores >= threshold).astype(int)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels,
        preds,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": threshold,
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "num_pairs": int(len(rows)),
        "num_positive_pairs": int(labels.sum()),
        "num_negative_pairs": int(len(rows) - labels.sum()),
    }


def save_ranked_csv(rows, output_path):
    by_table = defaultdict(list)
    label_by_pair = {}

    for row in rows:
        table_id = str(row["table_id"])
        text_id = str(row["text_id"])
        score = float(row["score"])
        label = int(row["label"])
        by_table[table_id].append((score, text_id))
        label_by_pair[(table_id, text_id)] = label

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["table_id", "text_id", "score", "rank", "label"],
        )
        writer.writeheader()
        for table_id in sorted(by_table):
            ranked = sorted(by_table[table_id], key=lambda x: (-x[0], x[1]))
            for rank, (score, text_id) in enumerate(ranked, start=1):
                writer.writerow({
                    "table_id": table_id,
                    "text_id": text_id,
                    "score": score,
                    "rank": rank,
                    "label": label_by_pair[(table_id, text_id)],
                })


def print_report(cls_report, rank_report, ks):
    print("=== Validation classification metrics ===")
    print(f"threshold: {cls_report['threshold']:.4f}")
    print(f"num_pairs: {cls_report['num_pairs']}")
    print(f"positive pairs: {cls_report['num_positive_pairs']}")
    print(f"negative pairs: {cls_report['num_negative_pairs']}")
    print(f"accuracy:  {cls_report['accuracy']:.4f}")
    print(f"precision: {cls_report['precision']:.4f}")
    print(f"recall:    {cls_report['recall']:.4f}")
    print(f"f1:        {cls_report['f1']:.4f}")

    print("\n=== Validation ranking metrics, macro-averaged over tables ===")
    for k in ks:
        m = rank_report[k]
        print(
            f"k={k:>3} | "
            f"P@k={m['P@k']:.4f} | "
            f"R@k={m['R@k']:.4f} | "
            f"F1@k={m['F1@k']:.4f} | "
            f"MRR@k={m['MRR@k']:.4f} | "
            f"#tables={m['num_tables']}"
        )
    print(f"MRR: {rank_report['MRR']:.4f}")


def main():
    ap = argparse.ArgumentParser(
        description="Evaluate validation predictions saved by tools/train_relation_classifier.py."
    )
    ap.add_argument(
        "--valid_predictions",
        required=True,
        help="Path to valid_preds.jsonl. Each row must contain table_id, text_id, label, score.",
    )
    ap.add_argument(
        "--k_eval",
        type=int,
        nargs="+",
        default=[1, 3, 5, 10, 20],
        help="k values for P@k/R@k/F1@k/MRR@k.",
    )
    ap.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary classification metrics.",
    )
    ap.add_argument(
        "--save_ranked_csv",
        default="",
        help="Optional path to save ranked validation predictions as CSV.",
    )
    args = ap.parse_args()

    rows = load_jsonl(args.valid_predictions)
    if not rows:
        raise ValueError(f"No rows found in {args.valid_predictions}")

    all_rankings, rel_by_table = build_rankings_and_gold(rows)
    cls_report = classification_metrics(rows, args.threshold)
    rank_report = macro_eval(all_rankings, rel_by_table, args.k_eval)

    print_report(cls_report, rank_report, args.k_eval)

    if args.save_ranked_csv:
        save_ranked_csv(rows, args.save_ranked_csv)
        print(f"\nsaved ranked predictions to: {args.save_ranked_csv}")


if __name__ == "__main__":
    main()
