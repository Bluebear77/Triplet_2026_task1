from typing import Dict, List
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    if logits.shape[-1] == 2:
        # 2-class classification metrics
        preds = np.argmax(logits, axis=-1)
        p, r, f1, _ = precision_recall_fscore_support(
            labels,
            preds,
            average="binary",
            zero_division=0,
        )
        acc = (preds == labels).mean()
        return {
            "accuracy": acc,
            "precision": p,
            "recall": r,
            "f1": f1,
        }
    elif logits.shape[-1] == 1:
        # regression metrics (using 0.5 threshold)
        logits, labels = eval_pred
        logits = np.squeeze(logits, axis=-1)
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

        precision, recall, f1, _ = precision_recall_fscore_support(
            labels, preds, average="binary", zero_division=0
        )
        acc = accuracy_score(labels, preds)

        return {
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    else:
        raise ValueError(f"Unexpected number of logits: {logits.shape[-1]}")

# -------------------------
# Evaluation (P@k, R@k, F1@k)
# -------------------------
def precision_at_k(ranked: List[str], relevant: set, k: int) -> float:
    top = ranked[:k]
    if k == 0:
        return 0.0
    return sum(1 for x in top if x in relevant) / k

def recall_at_k(ranked: List[str], relevant: set, k: int) -> float:
    if not relevant:
        return 0.0
    top = ranked[:k]
    return sum(1 for x in top if x in relevant) / len(relevant)

def f1(p: float, r: float) -> float:
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def reciprocal_rank(ranked: List[str], relevant: set) -> float:
    for i, x in enumerate(ranked):
        if x in relevant:
            return 1.0 / (i + 1)
    return 0.0

def macro_eval(all_rankings: Dict[str, List[str]], rel_by_table: Dict[str, set], ks: List[int]) -> Dict[int, Dict[str, float]]:
    out = {}
    for k in ks:
        ps, rs, fs, mrr= [], [], [], []
        for table_id, ranked in all_rankings.items():
            relevant = rel_by_table.get(table_id, set())
            p = precision_at_k(ranked, relevant, k)
            r = recall_at_k(ranked, relevant, k)
            rr = reciprocal_rank(ranked[:k], relevant)
            ps.append(p); rs.append(r); fs.append(f1(p, r)); mrr.append(rr)
        out[k] = {
            "P@k": sum(ps) / len(ps) if ps else 0.0,
            "R@k": sum(rs) / len(rs) if rs else 0.0,
            "F1@k": sum(fs) / len(fs) if fs else 0.0,
            "MRR@k": sum(mrr) / len(mrr) if mrr else 0.0,
            "num_tables": len(all_rankings),
        }
    out["MRR"] = sum(reciprocal_rank(all_rankings[tid], rel_by_table.get(tid, set())) for tid in all_rankings) / len(all_rankings) if all_rankings else 0.0

    return out


