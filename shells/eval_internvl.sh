#!/bin/bash

PARTITION=${PARTITION:-"llm_s"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-4}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

# 常量路径
OUTPUTS_DIR="outputs_v1_${GPUS}"
LOG_DIR="logs_v1_${GPUS}"

# 循环不同的数据集和答案文件
declare -a model_paths=( \
    'ckpts/OpenGVLab/InternVL-Chat-V1-5' \
)

declare -a tasks=( \
    'retrieval-text' \
    'retrieval-image' \
    'counting-text' \
    'counting-image' \
    'reasoning-text' \
    'reasoning-image' \
)

mkdir -p $LOG_DIR

for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#tasks[@]}; j++)); do
        model_path=${model_paths[i]}
        task=${tasks[j]}

        model_name="$(basename ${model_path})"

        srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            --job-name="eval_${model_name}_${task}" \
            python -u eval_internvl.py \
            --model-path $model_path \
            --task $task \
            --outputs-dir $OUTPUTS_DIR \
            --num-gpus-per-rank ${GPUS_PER_TASK} \
            2>&1 | tee -a "${LOG_DIR}/${model_name}_${task}.log"

        cat ${OUTPUTS_DIR}/temp_${model_name}_${task}/* > ${OUTPUTS_DIR}/${model_name}_${task}.jsonl
    done
done
