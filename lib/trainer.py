import torch
import torch.nn as nn
from transformers import Trainer

class WeightedTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        # metadata keys are already popped in collator during training batches
        outputs = model(**inputs)
        logits = outputs.logits

        if logits.shape[-1] > 1:
            if self.class_weights is not None:
                loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(logits.device))
            else:
                loss_fct = nn.CrossEntropyLoss()
        else:
            logits = outputs.logits.squeeze(-1)   # [B, 1] -> [B]
            if self.class_weights is not None:
                loss_fct = nn.BCEWithLogitsLoss(pos_weight=self.class_weights[1].to(logits.device))

        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
