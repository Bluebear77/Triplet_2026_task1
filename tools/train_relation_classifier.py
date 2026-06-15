import os
import torch.nn as nn
import argparse
import json

import torch
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    set_seed,
)

from lib.utils_task1 import split_by_table
from lib.data_pipeline import load_jsonl
from lib.eval_task1 import compute_metrics
from lib.trainer import WeightedTrainer

from lib.dataset_task1 import (
    RelationDataset,
    CollatorWithMeta,
)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--model_name", default="microsoft/deberta-v3-base")
    ap.add_argument("--output_dir", default="outputs/relation_classifier")
    ap.add_argument("--valid_ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_length", type=int, default=512)
    ap.add_argument("--train_batch_size", type=int, default=8)
    ap.add_argument("--eval_batch_size", type=int, default=16)
    ap.add_argument("--learning_rate", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--num_train_epochs", type=int, default=3)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--use_reranker", action="store_true")
    ap.add_argument("--use_instuctions", action="store_true", help="whether to use instructions in the input")
    ap.add_argument("--use_class_weight", action="store_true")
    ap.add_argument("--save_valid_predictions", default="")
    ap.add_argument("--resume_from_checkpoint", default="", help="Path to Trainer checkpoint to resume from")
    args = ap.parse_args()


    os.makedirs(args.output_dir, exist_ok=True)
    with open(f'{args.output_dir}/train_args.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    set_seed(args.seed)

    rows = load_jsonl(args.train_jsonl)

    train_rows, valid_rows = split_by_table(
        rows,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
    )

    print(f"train rows: {len(train_rows)}")
    print(f"valid rows: {len(valid_rows)}")
    print(f"train tables: {len(set(r['table_id'] for r in train_rows))}")
    print(f"valid tables: {len(set(r['table_id'] for r in valid_rows))}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2 if not args.use_reranker else 1,
        ignore_mismatched_sizes=True,
    )

    instruction = "Task: Determine whether the text is relevant to the table." if args.use_instuctions else ""

    train_dataset = RelationDataset(train_rows, tokenizer, instruction=instruction, use_float_labels=args.use_reranker, max_length=args.max_length)
    valid_dataset = RelationDataset(valid_rows, tokenizer, instruction=instruction, use_float_labels=args.use_reranker, max_length=args.max_length)
    collator = CollatorWithMeta(tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=100,
        load_best_model_at_end=True,
        fp16=args.fp16,
        report_to="none",
        save_total_limit=2,
    )

    if not args.use_class_weight:

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
        )

    else:
        num_pos = sum(int(r["label"]) for r in train_rows)
        num_neg = len(train_rows) - num_pos
        print(f"train positives: {num_pos}")
        print(f"train negatives: {num_neg}")

        # label 0 = unrelated, label 1 = related
        pos_weight = num_neg / max(num_pos, 1)
        class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)
        print(f"class weights: {class_weights.tolist()}")
        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            data_collator=collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
        )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint or None)

    metrics = trainer.evaluate()
    print("=== Validation metrics ===")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if args.save_valid_predictions:
        model.eval()
        device = model.device

        loader = DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=collator,
        )

        rows_out = []
        with torch.no_grad():
            for batch in loader:
                table_ids = batch.pop("table_id")
                text_ids = batch.pop("text_id")
                labels = batch.pop("labels")

                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                if not args.use_reranker:
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                else:
                    probs = torch.sigmoid(outputs.logits).squeeze(-1).cpu().numpy()

                for table_id, text_id, y, score in zip(
                    table_ids, text_ids, labels.tolist(), probs
                ):
                    rows_out.append({
                        "table_id": table_id,
                        "text_id": text_id,
                        "label": y,
                        "score": float(score),
                    })

        with open(args.save_valid_predictions, "w", encoding="utf-8") as f:
            for row in rows_out:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"saved validation predictions to: {args.save_valid_predictions}")


if __name__ == "__main__":
    main()