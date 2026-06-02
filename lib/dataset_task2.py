import torch
from torch.utils.data import Dataset

class RelationDataset(Dataset):
    def __init__(self, rows, tokenizer, max_length=512, no_labels=False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.no_labels = no_labels

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        triple_text, evidence_text = self.build_inputs(row)
        enc = self.tokenizer(
            triple_text,
            evidence_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )
        if not self.no_labels:
            enc["labels"] = int(row["label"])
        enc["document_id"] = row["document_id"]
        enc["subject"] = row["subject"]
        enc["object"] = row["object"]
        enc["predicate"] = row["predicate"]
        return enc

    def build_inputs(self, row: dict) -> tuple[str, str]:
        triple_text = (
            f"Subject: {row['subject']} "
            f"Predicate ID: {row['predicate']} "
            f"Predicate: {row['predicate_label']} "
            f"Object: {row['object']} "
            #f"description: {row['description']}"
        )

        evidence_text = (
            f"Title: {row['title']}\n"
            f"Table: {row['table']}\n"
            f"Text: {row['text']}"
        )
        return triple_text, evidence_text


class CollatorWithMeta:
    def __init__(self, tokenizer, no_labels=False):
        self.tokenizer = tokenizer
        self.no_labels = no_labels

    def __call__(self, features):
        doc_ids = [f.pop("document_id") for f in features]
        subjects = [f.pop("subject") for f in features]
        objects = [f.pop("object") for f in features]
        predicates = [f.pop("predicate") for f in features]
        
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        if not self.no_labels:
            labels = [f["labels"] for f in features]
            batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["document_id"] = doc_ids
        batch["subject"] = subjects
        batch["object"] = objects
        batch["predicate"] = predicates
        return batch
