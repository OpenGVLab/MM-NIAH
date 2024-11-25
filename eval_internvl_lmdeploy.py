
import os
import io
import json
import time
import argparse
import torch

from PIL import Image
from tqdm import tqdm

from transformers import AutoConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig

from utils.tools import get_input, init_dist, IMAGE_PLACEHOLDER, ConvertLMDeployChatTemplate

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

FILEPATH = 'shells/data/mm_niah.json'


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def save_outputs(outputs, results_file):
    world_size = torch.distributed.get_world_size()
    merged_outputs = [None for _ in range(world_size)]
    torch.distributed.all_gather_object(merged_outputs, outputs)

    merged_outputs = sum(merged_outputs, start=[])

    if torch.distributed.get_rank() == 0:
        os.makedirs(os.path.dirname(results_file), exist_ok=True)
        with open(results_file, 'a') as file:
            file.writelines(merged_outputs)

        print(f'[{current_time()}] Results ({len(merged_outputs)=}) saved to {results_file}')


def main(args):
    init_dist(args)

    task = args.task
    model_name = os.path.basename(args.model_path)
    results_file = os.path.join(args.outputs_dir, f'{model_name}_{task}.jsonl')

    if os.path.exists(results_file):
        print(f'{results_file} exists, early stop')
        exit(0)

    gen_config = GenerationConfig(
        temperature=0,
        max_new_tokens=64 if 'counting' in task else 16,
    )
    model_config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    chat_template_config = ConvertLMDeployChatTemplate(model_config.template)

    pipe = pipeline(
        args.model_path,
        chat_template_config=chat_template_config,
        backend_config=TurbomindEngineConfig(session_len=128000, cache_max_entry_count=0.1, tp=args.num_gpus_per_rank)
    )
    pipe.vl_encoder.model.config.dynamic_image_size = False

    torch.cuda.set_device(args.local_rank)
    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.model_path} on task {task}"
    )

    with open(args.question_file, 'r') as file:
        lines = file.readlines()

    lines = lines[args.rank::args.world_size]
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x:x['meta']['context_length'])

    oom_cnt = 0
    outputs_list = []
    for sample in tqdm(lines, desc=f"[{task}]", disable=args.rank!=0):
        if oom_cnt >= 20:
            print(f"[Rank {args.rank}] early stops because of successive failures. {oom_cnt=}")
            outputs_list.append(json.dumps({
                "question_id": sample['id'],
                "question": question,
                "answer": sample['answer'],
                "response": 'None',
                'context_length':sample['meta']['context_length'],
                'placed_depth':sample['meta']['placed_depth']
            }) + "\n")
            continue

        context, images_list, question, answer = get_input(sample)
        images_list = [os.path.join(args.image_folder, i) for i in images_list]
        images_list = [
            Image.open(image).convert('RGB')
            for image in images_list
        ]

        qs = f'{context}\n{question}'
        qs = qs.replace(IMAGE_PLACEHOLDER, IMAGE_TOKEN)

        try:
            outputs = pipe([(qs, images_list)], gen_config=gen_config)[0].text
            oom_cnt = 0
        except torch.cuda.OutOfMemoryError:
            print(f"[Rank {args.rank}] OutOfMemoryError occurs! totoal_tokens={sample['meta']['context_length']}")
            outputs = 'None'
            oom_cnt += 1
            torch.cuda.empty_cache()

        outputs = outputs.strip()
        print(f"[{current_time()}] [Rank {args.rank}] totoal_tokens={sample['meta']['context_length']}, {outputs=}")

        outputs_list.append(json.dumps({
            "question_id": sample['id'],
            "question": question,
            "answer": sample['answer'],
            "response": outputs,
            'context_length': sample['meta']['context_length'],
            'placed_depth': sample['meta']['placed_depth'],
        }) + "\n")

    print(f"[{current_time()}] Rank {args.rank} Finish")
    save_outputs(outputs_list, results_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for InternVL-Chat")
    parser.add_argument('--model-path', type=str, default='OpenGVLab/InternVL-Chat-V1-5')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--outputs-dir', type=str, default='')
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    args = parser.parse_args()

    with open(FILEPATH) as file:
        meta = json.load(file)

    args.image_folder = meta[args.task]['root']
    args.question_file = meta[args.task]['annotation']
    main(args)
