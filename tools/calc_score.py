import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import argparse

def score(gold, pred):
    # キーでマージ（順番ずれ対策）
    df = gold.merge(pred, on=["text_id", "table_id"], suffixes=("_gold", "_pred"))

    y_true = df["label_gold"]
    y_pred = df["label_pred"]

    # 指標
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    accuracy = accuracy_score(y_true, y_pred)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    precision_macro = precision_score(y_true, y_pred, average="macro")
    recall_macro = recall_score(y_true, y_pred, average="macro")
    f1_macro = f1_score(y_true, y_pred, average="macro")

    print(f"Macro F1: {f1_macro:.4f}")

    from sklearn.metrics import classification_report

    print(classification_report(y_true, y_pred, digits=4))


if __name__ == "__main__":

    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--pred", required=True)
    args = ap.parse_args()

    # 読み込み
    gold = pd.read_csv(args.gold)
    pred = pd.read_csv(args.pred)

    score(gold, pred)
