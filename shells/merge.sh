#!/bin/bash

OUTPUTS_DIR=$1

# 循环不同的数据集和答案文件
declare -a model_paths=( \
    'ckpts/liuhaotian/llava-v1.5-13b' \
    'ckpts/liuhaotian/llava-v1.6-vicuna-13b' \
    'ckpts/Efficient-Large-Model/VILA1.0-13b-llava' \
    'ckpts/liuhaotian/llava-v1.6-34b' \
    'ckpts/OpenGVLab/InternVL-Chat-V1-5' \
    'ckpts/BAAI/Emu2-Chat' \
)

declare -a tasks=( \
    # 'retrieval-text' \
    # 'retrieval-image' \
    # 'counting-text' \
    # 'counting-image' \
    # 'reasoning-text' \
    # 'reasoning-image' \
    'retrieval-text-v2' \
    'retrieval-image-v2' \
    'counting-text-v2' \
    'counting-image-v2' \
    'reasoning-text-v2' \
    'reasoning-image-v2' \
)

for ((i=0; i<${#model_paths[@]}; i++)); do
    for ((j=0; j<${#tasks[@]}; j++)); do
        model_path=${model_paths[i]}
        task=${tasks[j]}

        model_name="$(basename ${model_path})"

        cat ${OUTPUTS_DIR}/temp_${model_name}_${task}/* > ${OUTPUTS_DIR}/${model_name}_${task}.jsonl
        wc -l ${OUTPUTS_DIR}/${model_name}_${task}.jsonl
    done
done
