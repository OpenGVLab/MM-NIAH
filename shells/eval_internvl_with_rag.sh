#!/bin/bash

PARTITION=${PARTITION:-"llm_s"}
GPUS=${GPUS:-16}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
GPUS_PER_TASK=${GPUS_PER_TASK:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}

# 常量路径
OUTPUTS_DIR="outputs_v2_rag_${GPUS}"
LOG_DIR="logs_v2_rag_${GPUS}"

# 循环不同的数据集和答案文件
declare -a model_paths=( \
    'ckpts/OpenGVLab/InternVL-Chat-V1-5' \
)

declare -a tasks=( \
    # 'retrieval-text-rag-v2' \
    # 'retrieval-image-rag-v2' \
    'counting-text-rag-v2' \
    'counting-image-rag-v2' \
    # 'reasoning-text-rag-v2' \
    # 'reasoning-image-rag-v2' \
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
            -o "${LOG_DIR}/${model_name}_${task}.log" \
            -e "${LOG_DIR}/${model_name}_${task}.log" \
            --async \
            python -u eval_internvl.py \
            --model-path $model_path \
            --task $task \
            --outputs-dir $OUTPUTS_DIR \
            --num-gpus-per-rank ${GPUS_PER_TASK} \

        cat ${OUTPUTS_DIR}/temp_${model_name}_${task}/* > ${OUTPUTS_DIR}/${model_name}_${task}.jsonl
    done
done
