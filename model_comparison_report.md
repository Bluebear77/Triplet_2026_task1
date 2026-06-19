
# TRIPLET Challenge Subtask 1 - Model Comparison Report

For [../tools/train_relation_classifier.py](https://github.com/DensoITLab/tripletChallenge2026_task1/blob/main/tools/train_relation_classifier.py), a small bug (instruction should be controlled by `use_instuctions` NOT by `use_reranker`)is fixed by aligning instruction-style serialization with the design in Section 3.3, changing <br/>
`instruction = "Task: Determine whether the text is relevant to the table." if not args.use_reranker else None`
 to<br/>
 `instruction = "Task: Determine whether the text is relevant to the table." if args.use_instuctions else ""`.

## TRIPLET Challenge Combined Leaderboard (Official + Reproduced Results)

| Rank | Model / Participant | Submission ID | F1 Score | Precision | Recall | Accuracy |
|------|---------------------|---------------|----------|-----------|--------|----------|
| 1 | task1_fused_large_text__base_keyValue/alpha1.0 | — | **72.9** | 71.8 | 74.1 | **92.2** |
| 2 | task1_fused_large_text__base_text/alpha1.0 | — | 72.8 | 71.7 | 74.0 | 92.1 |
| 3 | **ytachioka (OFFICIAL SUBMISSION)** | **668767** | **72.0** | **74.0** | **69.0** | — |
| 4 | task1_roberta-large_text | — | 71.8 | **73.4** | 70.3 | 92.1 |
| 5 | task1_BAAI-bge-reranker-base_text | — | 70.4 | 66.8 | 74.4 | 91.1 |
| 6 | task1_roberta-base_text | — | 70.2 | 61.1 | 82.5 | 90.0 |
| 7 | task1_roberta-base_text_inst | — | 70.1 | 60.9 | 82.4 | 90.0 |
| 8 | task1_BAAI-bge-reranker-base_structured | — | 69.8 | 68.2 | 71.6 | 91.2 |
| 9 | task1_roberta-base_structured | — | 69.5 | 59.1 | **84.3** | 89.5 |
| 10 | task1_roberta-base_keyValue | — | 69.2 | 59.3 | 83.2 | 89.5 |

> **Note:** Official results were originally reported on a 0–1 scale and converted to percentage for consistency.

## Summary

**Best Model:** task1_fused_large_text__base_keyValue/alpha1.0

**Best F1 Score:** 0.7293362941031335

