import os
import io
import json
import argparse
import subprocess
import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchvision.transforms.functional import InterpolationMode

from utils.tools import get_input, init_dist

from petrel_client.client import Client
client = Client()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
FILEPATH = 'data.json'


def load_image(image_file):
    if 's3:' in image_file:
        data_bytes = client.get(image_file)
        assert data_bytes is not None, f'fail to load {image_file}'
        data_buff = io.BytesIO(data_bytes)
        image = Image.open(data_buff).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def build_model(args):
    num_gpus = torch.cuda.device_count()
    visible_devices = [i for i in range(args.local_rank, num_gpus, args.local_world_size)]

    if len(visible_devices) > 1:
        device_map = {}
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

        num_gpus_for_vit = 1
        num_gpus_for_llm = len(visible_devices) - num_gpus_for_vit

        num_layers = config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_llm
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu + num_gpus_for_vit, len(visible_devices) - 1)
            device_map[f'model.decoder.lm.model.layers.{i}'] = visible_devices[device_idx]

        num_layers = config.vision_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_vit
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu, num_gpus_for_vit - 1)
            device_map[f'model.visual.blocks.{i}'] = visible_devices[device_idx]

        device_map['model.visual.patch_embed.proj'] = 0
        device_map['model.visual.pos_embed'] = 0
        device_map['model.visual.cls_token'] = 0
        device_map['project_down.weight'] = num_gpus_for_vit - 1
        device_map['project_up.weight'] = num_gpus_for_vit - 1
        device_map['model.decoder.lm.model.embed_tokens.weight'] = num_gpus_for_vit
        device_map['model.decoder.lm.model.norm.weight'] = visible_devices[-1]
        device_map['model.decoder.lm.lm_head.weight'] = visible_devices[-1]

    else:
        device_map = {'': visible_devices[0]}

    if args.rank == 0:
        for k, v in device_map.items():
            print(k, v)

    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map=device_map,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    tokenizer.model_max_length = 256000

    return model, tokenizer


def main(args):
    init_dist(args)

    task = args.task
    model_name = os.path.basename(args.model_path)
    model, tokenizer = build_model(args)

    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.model_path} on task {task}, "
        f"devices: {set([p.device for p in model.parameters()])}"
    )

    ans_file_name = f'{model_name}_{task}.jsonl'
    if args.use_rag:
        ans_file_name = f'rag_{ans_file_name}'

    temp_dir = f"temp_{model_name}_{task}"
    ans_file_path = os.path.join(args.outputs_dir, temp_dir, f"{args.rank}_{args.world_size}_{ans_file_name}")

    os.makedirs(args.outputs_dir, exist_ok=True)
    os.makedirs(os.path.join(args.outputs_dir, temp_dir), exist_ok=True)

    with open(args.question_file, 'r') as file:
        lines = file.readlines()
    ans_file = open(ans_file_path, 'w')

    lines = lines[args.rank::args.world_size]
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x:x['meta']['context_length'])

    oom_cnt = 0
    for sample in tqdm(lines, desc=f"Processing {ans_file_name}", disable=args.rank!=0):
        context, images_list, question, answer = get_input(sample)
        context = context.replace('</s>', '')

        if oom_cnt >= 20:
            print(f"Rank {args.rank} early stops because of successive failures. {oom_cnt=}")
            ans_file.write(json.dumps({
                "question_id": sample['id'],
                "question": question,
                "answer": sample['answer'],
                "response": 'None',
                'total_tokens':sample['meta']['context_length'],
                'position':sample['meta']['placed_depth']
            }) + "\n")
            ans_file.flush()
            continue

        # TODO: rm
        images_list = [
            os.path.join('s3://public-dataset/OBELISC/raw-images', i[len('obelisc/'):]) if i.startswith('obelisc/') else i
            for i in images_list
        ]
        images_list = [
            os.path.join(args.image_folder, i) if 's3://' not in i else i
            for i in images_list
        ]

        if args.use_rag:
            from utils.rag import rag
            context, images_list = rag(context, images_list, question, 3000)
        qs = f'{context}\n{question}'

        inputs = model.build_input_ids(
            text=[qs],
            tokenizer=tokenizer,
            image=load_images(images_list),
        )

        try:
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                image=inputs['image'].to(torch.bfloat16),
                do_sample=False,
                num_beams=1,
                max_new_tokens=64 if 'counting' in task else 32,
            )
            oom_cnt = 0
        except torch.cuda.OutOfMemoryError:
            print(f"Rank {args.rank} OutOfMemoryError occurs! totoal_tokens={sample['meta']['context_length']}")
            outputs = 'None'
            oom_cnt += 1
            torch.cuda.empty_cache()

        outputs = outputs.strip()
        print(f"totoal_tokens={sample['meta']['context_length']}, {outputs=}")

        ans_file.write(json.dumps({
            "question_id": sample['id'],
            "question": question,
            "answer": sample['answer'],
            "response": outputs,
            'total_tokens':sample['meta']['context_length'],
            'position':sample['meta']['placed_depth']
        }) + "\n")
        ans_file.flush()

    print(f"Rank {args.rank} Finish")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for InternVL-1.5")
    parser.add_argument('--model-path', type=str, default='')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--outputs-dir', type=str, default='')
    parser.add_argument('--use-rag', action='store_true', default=False)
    parser.add_argument('--num-gpus-per-rank', type=int, default=2)
    args = parser.parse_args()

    with open(FILEPATH) as file:
        meta = json.load(file)

    args.image_folder = meta[args.task]['root']
    args.question_file = meta[args.task]['annotation']
    main(args)
