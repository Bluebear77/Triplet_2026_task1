import argparse
from lib.data_pipeline import load_jsonl
import numpy as np
from transformers import AutoTokenizer

def inspect_truncation(rows, tokenizer, max_length=512):
    pair_lens = []
    pair_lens_trunc = []
    num_truncated = 0

    text_lens = []
    table_lens = []

    for row in rows:
        text_input = f"Text:\n{row['text']}"
        table_input = f"Title: {row['title']}\nTable:\n{row['table']}"

        text_ids = tokenizer.encode(text_input, add_special_tokens=False)
        table_ids = tokenizer.encode(table_input, add_special_tokens=False)

        text_lens.append(len(text_ids))
        table_lens.append(len(table_ids))

        full = tokenizer(
            text_input,
            table_input,
            truncation=False,
            padding=False,
        )
        trunc = tokenizer(
            text_input,
            table_input,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        full_len = len(full["input_ids"])
        trunc_len = len(trunc["input_ids"])

        pair_lens.append(full_len)
        pair_lens_trunc.append(trunc_len)

        if full_len > max_length:
            num_truncated += 1

    def summarize(xs, name):
        xs = np.array(xs)
        print(f"{name}:")
        print(f"  mean={xs.mean():.1f}")
        print(f"  p50 ={np.percentile(xs, 50):.1f}")
        print(f"  p90 ={np.percentile(xs, 90):.1f}")
        print(f"  p95 ={np.percentile(xs, 95):.1f}")
        print(f"  max ={xs.max()}")

    summarize(text_lens, "text_len")
    summarize(table_lens, "table_len")
    summarize(pair_lens, "pair_len_before_trunc")
    summarize(pair_lens_trunc, "pair_len_after_trunc")

    print(f"truncated: {num_truncated}/{len(rows)} ({num_truncated/len(rows):.2%})")

def inspect_truncation_loss(rows, tokenizer, max_length=512):
    losses = []

    for row in rows:
        text_input = f"Text:\n{row['text']}"
        table_input = f"Title: {row['title']}\nTable:\n{row['table']}"

        full = tokenizer(text_input, table_input, truncation=False, padding=False)
        full_len = len(full["input_ids"])

        lost = max(0, full_len - max_length)
        losses.append(lost)

    losses = np.array(losses)
    print(f"avg lost tokens: {losses.mean():.1f}")
    print(f"p90 lost tokens: {np.percentile(losses, 90):.1f}")
    print(f"max lost tokens: {losses.max()}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--max_length", type=int, default=512)
    args = ap.parse_args()

    rows = load_jsonl(args.train_jsonl)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    inspect_truncation(rows, tokenizer, max_length=args.max_length)
    inspect_truncation_loss(rows, tokenizer, max_length=args.max_length)
