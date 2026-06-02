import pandas as pd
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument("--threshold", type=float, default=0.5, help="Score threshold to determine label (default: 0.5)")
argparser.add_argument("--input_csv", required=True, help="Path to input CSV with columns: text_id, table_id, score, rank")
argparser.add_argument("--rank_based", action="store_true", help="Whether to derive labels based on rank (rank==1 -> label=1, else 0) instead of score (score > 0.5 -> label=1, else 0)")
argparser.add_argument("--output_csv", required=True, help="Path to output CSV with columns: text_id, table_id, label")
args = argparser.parse_args()

print(f"Reading input from {args.input_csv}...")

if args.input_csv.endswith(".jsonl"):
    with open(args.input_csv, "r", encoding="utf-8") as f:
        df = pd.read_json(f, lines=True)
else:
    df = pd.read_csv(args.input_csv)

if args.rank_based:
    df["label"] = (df["rank"] == 1).astype(int)
else:
    df["label"] = (df["score"] > args.threshold).astype(int)

# text_idでソート
df = df.sort_values(by="text_id")

out = df[["text_id", "table_id", "label"]]
out.to_csv(args.output_csv, index=False)
