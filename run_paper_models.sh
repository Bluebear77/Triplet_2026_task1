#!/usr/bin/env bash
set -Eeuo pipefail

# Run all model/serialization variants reported in the TRIPLET 2026 paper.
# Output is appended to record.txt by default.
LOG_FILE="${LOG_FILE:-record.txt}"
exec > >(tee -a "$LOG_FILE") 2>&1

# Configurable knobs.
MAX_LENGTH="${MAX_LENGTH:-512}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-1e-5}"
K_ROWS="${K_ROWS:-5}"
THRESHOLD="${THRESHOLD:-0.5}"
BASE_BS="${BASE_BS:-128}"        # README comment: 128 for roberta-base
LARGE_BS="${LARGE_BS:-46}"       # README comment: 46 for roberta-large
RERANKER_BS="${RERANKER_BS:-46}"
FORCE_PREPROCESS="${FORCE_PREPROCESS:-0}"
FORCE_TRAIN="${FORCE_TRAIN:-0}"
FORCE_SCORE="${FORCE_SCORE:-0}"

# Avoid DeepSpeed/Triton cache on NFS if possible.
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER:-user}/triton_autotune}"
mkdir -p "$TRITON_CACHE_DIR"

sanitize_model_name() {
  echo "$1" | sed 's|/|-|g'
}

latest_checkpoint_step() {
  local model_dir="$1"
  if [[ ! -d "$model_dir" ]]; then
    return 1
  fi
  find "$model_dir" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f\n' \
    | awk -F- '{print $2}' \
    | sort -n \
    | tail -n1
}

preprocess_split() {
  local split="$1"
  local table_type="$2"
  local out_jsonl="data/${split}/rel_data_${table_type}.jsonl"
  local test_opt=()

  if [[ "$split" == "test" ]]; then
    test_opt=(--test)
  fi

  if [[ "$FORCE_PREPROCESS" == "1" || ! -s "$out_jsonl" ]]; then
    echo ""
    echo "===== Preprocess ${split} / ${table_type}: $(date) ====="
    python -m tools.build_relation_dataset \
      --tables_path "data/${split}/tables.json" \
      --texts_path "data/${split}/texts.json" \
      --pairs_path "data/${split}/pairs.csv" \
      --table_input_type "$table_type" \
      --k_rows "$K_ROWS" \
      "${test_opt[@]}" \
      --output_jsonl "$out_jsonl"
  else
    echo "Skip preprocess ${split}/${table_type}; found $out_jsonl"
  fi
}

train_variant() {
  local run_name="$1"
  local model_name="$2"
  local table_type="$3"
  local train_bs="$4"
  local eval_bs="$5"
  local model_opt="$6"
  local inst_opt="$7"

  local train_jsonl="data/train/rel_data_${table_type}.jsonl"
  local out_base="outputs/train/task1_${run_name}"
  local model_dir="${out_base}/relation_cls"
  local latest=""
  latest="$(latest_checkpoint_step "$model_dir" || true)"

  if [[ "$FORCE_TRAIN" != "1" && -n "$latest" ]]; then
    echo "Skip train ${run_name}; found checkpoint-${latest}"
    return 0
  fi

  echo ""
  echo "===== Train ${run_name}: $(date) ====="
  mkdir -p "$out_base"

  local extra=()
  [[ -n "$model_opt" ]] && extra+=("$model_opt")
  [[ -n "$inst_opt" ]] && extra+=("$inst_opt")

  python -m tools.train_relation_classifier \
    --train_jsonl "$train_jsonl" \
    --model_name "$model_name" \
    --output_dir "$model_dir" \
    --max_length "$MAX_LENGTH" \
    --train_batch_size "$train_bs" \
    --eval_batch_size "$eval_bs" \
    --learning_rate "$LR" \
    --use_class_weight \
    --num_train_epochs "$EPOCHS" \
    "${extra[@]}" \
    --save_valid_predictions "${out_base}/valid_preds.jsonl"
}

score_variant() {
  local run_name="$1"
  local model_name="$2"
  local table_type="$3"
  local eval_bs="$4"
  local inst_opt="$5"

  local test_jsonl="data/test/rel_data_${table_type}.jsonl"
  local train_model_dir="outputs/train/task1_${run_name}/relation_cls"
  local latest=""
  latest="$(latest_checkpoint_step "$train_model_dir" || true)"

  if [[ -z "$latest" ]]; then
    echo "ERROR: no checkpoint found for ${run_name} under ${train_model_dir}" >&2
    exit 1
  fi

  local out_base="outputs/test/task1_${run_name}"
  local score_file="${out_base}/predicted_scores.jsonl"
  mkdir -p "$out_base"

  if [[ "$FORCE_SCORE" == "1" || ! -s "$score_file" ]]; then
    echo ""
    echo "===== Score ${run_name} using checkpoint-${latest}: $(date) ====="
    local extra=()
    [[ -n "$inst_opt" ]] && extra+=("$inst_opt")

    python -m tools.score_rel \
      --data_jsonl_path "$test_jsonl" \
      --model_name "$model_name" \
      --model_dir "${train_model_dir}/checkpoint-${latest}" \
      --max_length "$MAX_LENGTH" \
      --batch_size "$eval_bs" \
      "${extra[@]}" \
      --save_scores_jsonl "$score_file"
  else
    echo "Skip score ${run_name}; found $score_file"
  fi
}

make_submission() {
  local score_file="$1"
  local out_dir="$(dirname "$score_file")"

  echo ""
  echo "===== Make submission from ${score_file}: $(date) ====="
  python -m tools.format_res \
    --threshold "$THRESHOLD" \
    --input_csv "$score_file" \
    --output_csv "${out_dir}/predictions.csv"

  rm -f "${out_dir}/submission.zip"
  zip -j "${out_dir}/submission.zip" "${out_dir}/predictions.csv"
  ls -lh "${out_dir}/submission.zip"
}

run_variant() {
  local run_name="$1"
  local model_name="$2"
  local table_type="$3"
  local train_bs="$4"
  local eval_bs="$5"
  local model_opt="$6"
  local inst_opt="$7"

  preprocess_split train "$table_type"
  preprocess_split test "$table_type"
  train_variant "$run_name" "$model_name" "$table_type" "$train_bs" "$eval_bs" "$model_opt" "$inst_opt"
  score_variant "$run_name" "$model_name" "$table_type" "$eval_bs" "$inst_opt"
  make_submission "outputs/test/task1_${run_name}/predicted_scores.jsonl"
}

fuse_and_submit() {
  local fuse_name="$1"
  local score1="$2"
  local score2="$3"
  local alpha="$4"
  local out_dir="outputs/test/task1_fused_${fuse_name}/alpha${alpha}"
  local out_score="${out_dir}/predicted_scores.jsonl"

  if [[ ! -s "$score1" || ! -s "$score2" ]]; then
    echo "Skip fusion ${fuse_name}; missing input score file(s)."
    return 0
  fi

  echo ""
  echo "===== Fuse ${fuse_name}, alpha=${alpha}: $(date) ====="
  mkdir -p "$out_dir"
  python -m tools.fuse_scores \
    --score1 "$score1" \
    --score2 "$score2" \
    --pairs_path data/test/pairs.csv \
    --alpha "$alpha" \
    --save_scores_jsonl "$out_score"

  make_submission "$out_score"
}

main() {
  echo ""
  echo "===== RUN PAPER MODELS START: $(date) ====="
  echo "LOG_FILE=$LOG_FILE"
  echo "MAX_LENGTH=$MAX_LENGTH EPOCHS=$EPOCHS LR=$LR K_ROWS=$K_ROWS THRESHOLD=$THRESHOLD"
  echo "BASE_BS=$BASE_BS LARGE_BS=$LARGE_BS RERANKER_BS=$RERANKER_BS"
  echo "FORCE_PREPROCESS=$FORCE_PREPROCESS FORCE_TRAIN=$FORCE_TRAIN FORCE_SCORE=$FORCE_SCORE"

  # Variants reported in Table 1 of the paper.
  # Format: run_name|model_name|table_input_type|train_bs|eval_bs|model_opt|inst_opt
  local variants=(
    "roberta-large_text|roberta-large|text|$LARGE_BS|$LARGE_BS||"
    "roberta-base_text|roberta-base|text|$BASE_BS|$BASE_BS||"
    "roberta-base_text_inst|roberta-base|text|$BASE_BS|$BASE_BS||--use_instuctions"
    "roberta-base_structured|roberta-base|structured|$BASE_BS|$BASE_BS||"
    "roberta-base_keyValue|roberta-base|keyValue|$BASE_BS|$BASE_BS||"
    "BAAI-bge-reranker-base_text|BAAI/bge-reranker-base|text|$RERANKER_BS|$RERANKER_BS|--use_reranker|"
    "BAAI-bge-reranker-base_structured|BAAI/bge-reranker-base|structured|$RERANKER_BS|$RERANKER_BS|--use_reranker|"
  )

  local spec run_name model_name table_type train_bs eval_bs model_opt inst_opt
  for spec in "${variants[@]}"; do
    IFS='|' read -r run_name model_name table_type train_bs eval_bs model_opt inst_opt <<< "$spec"
    run_variant "$run_name" "$model_name" "$table_type" "$train_bs" "$eval_bs" "$model_opt" "$inst_opt"
  done

  # Extra candidate submissions. The paper does not specify these as required;
  # they are useful because the README already supports score fusion, and the
  # paper identifies roberta-large/text and roberta-base/keyValue as complementary.
  fuse_and_submit \
    "large_text__base_text" \
    "outputs/test/task1_roberta-large_text/predicted_scores.jsonl" \
    "outputs/test/task1_roberta-base_text/predicted_scores.jsonl" \
    "1.0"

  fuse_and_submit \
    "large_text__base_keyValue" \
    "outputs/test/task1_roberta-large_text/predicted_scores.jsonl" \
    "outputs/test/task1_roberta-base_keyValue/predicted_scores.jsonl" \
    "1.0"

  echo ""
  echo "===== ALL SUBMISSION FILES ====="
  find outputs/test -path '*/submission.zip' -type f -print -exec ls -lh {} \;
  echo "===== RUN PAPER MODELS END: $(date) ====="
}

main "$@"
