#!/bin/bash

preprocess() {
     python -m tools.build_relation_dataset \
     --tables_path data/${dtype}/tables.json \
     --texts_path data/${dtype}/texts.json \
     --pairs_path data/${dtype}/pairs.csv \
     --table_input_type ${table_input_type} \
     --k_rows 5 ${opt_args} \
     --output_jsonl data/${dtype}/rel_data.jsonl

}

train_classifier() {

  mkdir -p outputs/${dtype}/task1_${model0}${suffix}
  python -m tools.train_relation_classifier \
    --train_jsonl data/${dtype}/rel_data.jsonl \
    --model_name ${model}  \
    --output_dir outputs/${dtype}/task1_${model0}${suffix}/relation_cls \
    --max_length 512 \
    --train_batch_size ${train_batch_size} \
    --eval_batch_size ${eval_batch_size} \
    --learning_rate 1e-5 \
    --use_class_weight \
    --num_train_epochs 10 ${model_opt} ${inst_opt} \
    --save_valid_predictions outputs/${dtype}/task1_${model0}${suffix}/valid_preds.jsonl
}

eval_classifier() {

  mkdir -p outputs/${dtype}/task1_${model0}${suffix}
  local model_p_dir=$(echo outputs/${dtype}/task1_${model0}${suffix} | sed 's|/test|/train|')/relation_cls
  local latest_cp=$(ls ${model_p_dir}/ | grep checkpoint | awk -F- '{print $2}' | sort -n | tail -n1)

  python -m tools.score_rel \
    --data_jsonl_path data/${dtype}/rel_data.jsonl \
    --model_name ${model}  \
    --model_dir ${model_p_dir}/checkpoint-${latest_cp} \
    --max_length 512 \
    --batch_size ${eval_batch_size} ${inst_opt} \
    --save_scores_jsonl outputs/${dtype}/task1_${model0}${suffix}/predicted_scores.jsonl
}


make_submission() {
     local ranking_file=$1
     local threshold=$2
     local res_dir=$(dirname $ranking_file)
     python -m tools.format_res --threshold ${threshold} --input_csv ${ranking_file} --output_csv ${res_dir}/predictions.csv
     zip -j ${res_dir}/submission.zip ${res_dir}/predictions.csv
}




fuse_scores() {
     local score1_path=$1
     local score2_path=$2
     local alpha=$3
     mkdir -p outputs/${dtype}/task1_fused/alpha${alpha}
     python -m tools.fuse_scores \
          --score1 $score1_path \
          --score2 $score2_path \
          --pairs_path data/${dtype}/pairs.csv \
          --alpha $alpha \
          --save_scores_jsonl outputs/${dtype}/task1_fused/alpha${alpha}/predicted_scores.jsonl
}


model=roberta-base
#model="cross-encoder/ms-marco-MiniLM-L-6-v2"
#model="BAAI/bge-reranker-base"
#model_opt="--use_reranker"
model0=$(echo $model | sed 's|/|-|g')

train_batch_size=46  # 128 for roberta-base, 46 for roberta-large
eval_batch_size=${train_batch_size}

table_input_type="text"

suffix="_${table_input_type}"

dtype=train
opt_args=""
#inst_opt="--use_instuctions"
#suffix="${suffix}_inst"

preprocess

train_classifier

dtype=test
opt_args="--test"

preprocess

eval_classifier
make_submission outputs/${dtype}/task1_${model0}${suffix}/predicted_scores.jsonl 0.5

fuse_scores outputs/${dtype}/task1_roberta-large${suffix}/predicted_scores.jsonl \
     outputs/${dtype}/task1_roberta-base${suffix}/predicted_scores.jsonl 1.0
make_submission outputs/${dtype}/task1_fused/alpha1.0/predicted_scores.jsonl 0.5


