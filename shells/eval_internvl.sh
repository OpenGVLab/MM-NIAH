#!/bin/bash

PARTITION=${PARTITION:-"Intern5"}
GPUS=${GPUS:-64}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

OUTPUTS_DIR="outputs"
LOG_DIR="logs"

declare -a model_paths=( \
    'OpenGVLab/InternVL-Chat-V1-5' \
)

declare -a tasks=( \
    'retrieval-text-val' \
    'retrieval-image-val' \
    'counting-text-val' \
    'counting-image-val' \
    'reasoning-text-val' \
    'reasoning-image-val' \
    'retrieval-text-test' \
    'retrieval-image-test' \
    'counting-text-test' \
    'counting-image-test' \
    'reasoning-text-test' \
    'reasoning-image-test' \
)

mkdir -p $LOG_DIR

for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#tasks[@]}; j++)); do
        model_path=${model_paths[i]}
        task=${tasks[j]}

        model_name="$(basename ${model_path})"
        echo "$(date) ${model_name}_${task}"

        srun -p ${PARTITION} \
            --gres=gpu:${GPUS_PER_NODE} \
            --ntasks=$((GPUS / GPUS_PER_TASK)) \
            --ntasks-per-node=$((GPUS_PER_NODE / GPUS_PER_TASK)) \
            --quotatype=${QUOTA_TYPE} \
            --job-name="eval_${model_name}_${task}" \
            -o "${LOG_DIR}/${model_name}_${task}.log" \
            -e "${LOG_DIR}/${model_name}_${task}.log" \
            python -u eval_internvl.py \
            --model-path $model_path \
            --task $task \
            --outputs-dir $OUTPUTS_DIR \
            --num-gpus-per-rank ${GPUS_PER_TASK} \

        cat ${OUTPUTS_DIR}/temp_${model_name}_${task}/* > ${OUTPUTS_DIR}/${model_name}_${task}.jsonl
    done
done
