import torch
from torch.utils.data import Dataset

class RelationDataset(Dataset):
    def __init__(self, rows, tokenizer, instruction: str="", max_length=512, no_labels=False, use_float_labels=False):
        self.rows = rows
        self.tokenizer = tokenizer
        self.instruction = instruction

        self.max_length = max_length
        self.no_labels = no_labels
        self.use_float_labels = use_float_labels

    def __len__(self):
        return len(self.rows)



    def __getitem__(self, idx):

        def truncate_text_for_pair(tokenizer, text, max_tokens):
            enc = tokenizer(
                text,
                add_special_tokens=False,
                truncation=True,
                max_length=max_tokens,
            )
            return tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
        
      
        row = self.rows[idx]
        query_text, table_text = self.build_inputs(row)

        # query側の暴走を防ぐ
        query_text = truncate_text_for_pair(self.tokenizer, query_text, max_tokens=220)

        enc = self.tokenizer(
            query_text,
            table_text,
            truncation="only_second", # we want to keep the full query if possible, and truncate the table if needed
            max_length=self.max_length,
            padding=False,
        )

        if not self.no_labels:
            enc["labels"] = float(row["label"]) if self.use_float_labels else int(row["label"])
        enc["table_id"] = row["table_id"]
        enc["text_id"] = row["text_id"]
        return enc


    def build_inputs(self, row: dict) -> tuple[str, str]:
        query_text = (
            (self.instruction + "\n") if self.instruction else ""
        ) + f"Text:\n{row['text']}"

        table_text = (
            f"Title: {row['title']}\n"
            f"Table:\n{row['table']}\n"
        )
        return query_text, table_text


class CollatorWithMeta:
    def __init__(self, tokenizer, no_labels=False):
        self.tokenizer = tokenizer
        self.no_labels = no_labels

    def __call__(self, features):
        table_ids = [f.pop("table_id") for f in features]
        text_ids = [f.pop("text_id") for f in features]
        
        batch = self.tokenizer.pad(
            features,
            padding=True,
            return_tensors="pt",
        )
        if not self.no_labels:
            labels = [f["labels"] for f in features]
            if isinstance(labels[0], float):
                batch["labels"] = torch.tensor(labels, dtype=torch.float)
            else:
                batch["labels"] = torch.tensor(labels, dtype=torch.long)
        batch["table_id"] = table_ids
        batch["text_id"] = text_ids
        return batch
