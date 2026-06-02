import os
import argparse
import json
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from lib.data_pipeline import (
    load_jsonl,
)

from lib.dataset_task1 import (
    RelationDataset,
    CollatorWithMeta,
)

@torch.no_grad()
def score_rows(model, tokenizer, rows, max_length=512, batch_size=8, use_instructions=False):
    dataset = RelationDataset(rows, tokenizer, 
                              instruction='' if not use_instructions else "Task: Determine whether the text is relevant to the table.",
                              max_length=max_length, no_labels=True)
    collator = CollatorWithMeta(tokenizer, no_labels=True)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    model.eval()
    device = model.device

    scored = []
    for batch in loader:
        table_ids = batch.pop("table_id")
        text_ids = batch.pop("text_id")

        inputs = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**inputs)
        if outputs.logits.shape[-1] > 1:
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].detach().cpu().numpy()
        else:
            probs = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()

        for table_id, text_id, score in zip(table_ids, text_ids, probs):
            scored.append({
                "table_id": table_id,
                "text_id": text_id,
                "score": float(score),
            })
    return scored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_jsonl_path", required=True)
    ap.add_argument("--model_name", default="roberta-base")
    ap.add_argument("--model_dir", required=True, help="directory with trained classifier checkpoint")

    ap.add_argument("--use_instuctions", action="store_true", help="whether to use instructions in the input")
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--save_scores_jsonl", required=True, help="path to save scores in JSONL format")
    args = ap.parse_args()

    with open(f'{os.path.dirname(args.save_scores_jsonl)}/test_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    rows = load_jsonl(args.data_jsonl_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    all_scores = []

    scored = score_rows(
        model,
        tokenizer,
        rows,
        max_length=args.max_length,
        batch_size=args.batch_size,
        use_instructions=args.use_instuctions,
    )

    all_scores.extend(scored)

    print(f"num predicted scores: {len(all_scores)}")

    with open(args.save_scores_jsonl, "w", encoding="utf-8") as f:
        for r in all_scores:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"saved scores to: {args.save_scores_jsonl}")


if __name__ == "__main__":
    main()
