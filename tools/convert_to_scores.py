import json
import csv
import argparse

from task1_bm25_baseline import (
    load_pairs_csv,
)

ap = argparse.ArgumentParser()
ap.add_argument("--output_choices_path", required=True, help="Path to output choices JSON file containing predictions.")
ap.add_argument("--pairs_path", required=True)
args = ap.parse_args()

with open(args.output_choices_path, "r", encoding="utf-8") as f:
    data = json.load(f)

pairs = load_pairs_csv(args.pairs_path)

res = []
for item in data:

    table_id = pairs[int(item["idx"])][1]
    text_id = pairs[int(item["idx"])][0]    
    #res.append((table_id, text_id, item["predict"]["related"]))
    res.append((table_id, text_id, (item["predict"]["related"]-item["predict"]["unrelated"])))

# resをtable_idでソート
res.sort(key=lambda x: x[0])


with open("output_choices.csv", "w", encoding="utf-8", newline="") as f:
    w = csv.writer(f)
    w.writerow(["table_id", "text_id", "score"])
    w.writerows(res)
