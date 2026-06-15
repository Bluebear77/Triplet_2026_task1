# Candidate-Aware Table Serialization and Cross-Encoder Ranking for Table--Text Relatedness

This repository contains the system submitted to **TRIPLET Challenge @ ESWC 2026 - Subtask 1: Relatedness**.

The system builds table--text relatedness examples from the provided tables, texts, and candidate pairs, serializes tables into textual input, trains a Transformer-based relation classifier, and ranks candidate pairs by predicted relatedness scores.

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{tachioka2026triplet,
  author = {Y. Tachioka},
  title = {Candidate-Aware Table Serialization and Cross-Encoder Ranking for Table--Text Relatedness},
  booktitle = {TRIPLET Workshop 2026, co-located with ESWC 2026},
  year = {2026}
}
```

## Overview

This repository addresses table--text relatedness prediction for TRIPLET Challenge Task 1.

The pipeline consists of:

1. Building relation-classification datasets from tables, texts, and candidate pairs.
2. Serializing tables into textual input.
3. Training a Transformer-based relation classifier.
4. Scoring candidate table--text pairs.
5. Formatting predictions for Codabench submission.
6. Optionally fusing scores from multiple model outputs.

The default configuration in `run.sh` uses `roberta-base`.

## Repository Structure

```text
.
├── data/
│   ├── train/
│   │   ├── tables.json
│   │   ├── texts.json
│   │   ├── pairs.csv
│   │   └── rel_data.jsonl
│   └── test/
│       ├── tables.json
│       ├── texts.json
│       ├── pairs.csv
│       └── rel_data.jsonl
├── outputs/
│   ├── train/
│   └── test/
├── tools/
│   ├── build_relation_dataset.py
│   ├── train_relation_classifier.py
│   ├── score_rel.py
│   ├── format_res.py
│   └── fuse_scores.py
├── run.sh
├── requirements.txt
└── README.md
```

## Environment

The Python environment used for the submitted experiments was exported with `pip freeze`.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

The environment includes, among others:

- `torch==2.6.0+cu124`
- `transformers==5.3.0`
- `sentence-transformers==5.2.3`
- `accelerate==1.13.0`
- `datasets==4.0.0`
- `pandas==2.3.1`
- `scikit-learn==1.7.2`

Depending on your CUDA and PyTorch setup, you may need to adjust the PyTorch installation command for your local environment.

## Data Preparation

Place the official TRIPLET Challenge Task 1 files under `data/train/` and `data/test/`.

Expected input files are:

```text
data/train/tables.json
data/train/texts.json
data/train/pairs.csv

data/test/tables.json
data/test/texts.json
data/test/pairs.csv
```

The preprocessing step creates:

```text
data/train/rel_data.jsonl
data/test/rel_data.jsonl
```

## Running the Full Pipeline

Run:

```bash
bash run.sh
```

The script performs preprocessing, training, inference, submission formatting, and optional score fusion.

## Pipeline Details

### 1. Preprocess Training Data

```bash
python -m tools.build_relation_dataset \
  --tables_path data/train/tables.json \
  --texts_path data/train/texts.json \
  --pairs_path data/train/pairs.csv \
  --table_input_type text \
  --k_rows 5 \
  --output_jsonl data/train/rel_data.jsonl
```

### 2. Train Relation Classifier

```bash
python -m tools.train_relation_classifier \
  --train_jsonl data/train/rel_data.jsonl \
  --model_name roberta-base \
  --output_dir outputs/train/task1_roberta-base_text/relation_cls \
  --max_length 512 \
  --train_batch_size 46 \
  --eval_batch_size 46 \
  --learning_rate 1e-5 \
  --use_class_weight \
  --num_train_epochs 10 \
  --save_valid_predictions outputs/train/task1_roberta-base_text/valid_preds.jsonl
```

### 3. Preprocess Test Data

```bash
python -m tools.build_relation_dataset \
  --tables_path data/test/tables.json \
  --texts_path data/test/texts.json \
  --pairs_path data/test/pairs.csv \
  --table_input_type text \
  --k_rows 5 \
  --test \
  --output_jsonl data/test/rel_data.jsonl
```

### 4. Score Test Pairs

The script automatically selects the latest checkpoint under the corresponding training output directory.

```bash
python -m tools.score_rel \
  --data_jsonl_path data/test/rel_data.jsonl \
  --model_name roberta-base \
  --model_dir outputs/train/task1_roberta-base_text/relation_cls/checkpoint-<STEP> \
  --max_length 512 \
  --batch_size 46 \
  --save_scores_jsonl outputs/test/task1_roberta-base_text/predicted_scores.jsonl
```

### 5. Create Submission File

```bash
python -m tools.format_res \
  --threshold 0.5 \
  --input_csv outputs/test/task1_roberta-base_text/predicted_scores.jsonl \
  --output_csv outputs/test/task1_roberta-base_text/predictions.csv

zip -j outputs/test/task1_roberta-base_text/submission.zip \
  outputs/test/task1_roberta-base_text/predictions.csv
```

The final Codabench submission file is:

```text
outputs/test/task1_roberta-base_text/submission.zip
```

## Model Configuration

The main configuration variables are defined in `run.sh`.

```bash
model=roberta-base
model0=$(echo $model | sed 's|/|-|g')

train_batch_size=46
eval_batch_size=${train_batch_size}

table_input_type="text"
suffix="_${table_input_type}"
```

### Standard Cross-Encoder / Sequence-Classification Models

For models such as:

```bash
model="cross-encoder/ms-marco-MiniLM-L-6-v2"
```

use the standard classifier path without `--use_reranker`:

```bash
model="cross-encoder/ms-marco-MiniLM-L-6-v2"
model0=$(echo $model | sed 's|/|-|g')
model_opt=""
```

### Reranker Models

For reranker-style models such as:

```bash
model="BAAI/bge-reranker-base"
```

enable reranker mode:

```bash
model="BAAI/bge-reranker-base"
model0=$(echo $model | sed 's|/|-|g')
model_opt="--use_reranker"
```

## Score Fusion

The script also supports fusing scores from two model outputs.

Example:

```bash
python -m tools.fuse_scores \
  --score1 outputs/test/task1_roberta-large_text/predicted_scores.jsonl \
  --score2 outputs/test/task1_roberta-base_text/predicted_scores.jsonl \
  --pairs_path data/test/pairs.csv \
  --alpha 1.0 \
  --save_scores_jsonl outputs/test/task1_fused/alpha1.0/predicted_scores.jsonl
```

Then create a submission file from the fused scores:

```bash
python -m tools.format_res \
  --threshold 0.5 \
  --input_csv outputs/test/task1_fused/alpha1.0/predicted_scores.jsonl \
  --output_csv outputs/test/task1_fused/alpha1.0/predictions.csv

zip -j outputs/test/task1_fused/alpha1.0/submission.zip \
  outputs/test/task1_fused/alpha1.0/predictions.csv
```

The fused submission file is:

```text
outputs/test/task1_fused/alpha1.0/submission.zip
```

## Outputs

Typical output files include:

```text
outputs/train/task1_roberta-base_text/relation_cls/
outputs/train/task1_roberta-base_text/valid_preds.jsonl
outputs/test/task1_roberta-base_text/predicted_scores.jsonl
outputs/test/task1_roberta-base_text/predictions.csv
outputs/test/task1_roberta-base_text/submission.zip
```

For fused predictions:

```text
outputs/test/task1_fused/alpha1.0/predicted_scores.jsonl
outputs/test/task1_fused/alpha1.0/predictions.csv
outputs/test/task1_fused/alpha1.0/submission.zip
```

## Notes

- The default table serialization mode is `text`.
- The preprocessing script uses up to `k_rows=5` table rows.
- The classifier input length is capped at `max_length=512`.
- Class weighting is enabled during training with `--use_class_weight`.
- The submission threshold is set to `0.5` by default.
- `cross-encoder/ms-marco-MiniLM-L-6-v2` should be run without `--use_reranker`.
- `BAAI/bge-reranker-base` should be run with `--use_reranker`.
- The final submission archive should contain `predictions.csv`.


## Command to reproduce:

```
#In circa01
cd  /home/wei/Triplet_2026_task1

conda create -p /home/wei/Triplet_2026_task1/.conda_env python=3.11 -y
conda activate /home/wei/Triplet_2026_task1/.conda_env

which python
python --version

conda install -c conda-forge zip -y

python -m pip install --upgrade pip


python -m pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 \
  --index-url https://download.pytorch.org/whl/cu124

grep -vE '^(torch|torchvision|torchaudio)==' requirements.txt > requirements_no_torch.txt
python -m pip install -r requirements_no_torch.txt

python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"

ls data/train
ls data/test

export CUDA_VISIBLE_DEVICES=2
bash run.sh 2>&1 | tee record.txt
```