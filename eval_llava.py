import os
import io
import json
import argparse
import subprocess
import torch

from PIL import Image
from tqdm import tqdm
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.mm_utils import (
    process_images,
    tokenizer_image_token,
    get_model_name_from_path,
)
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates

from utils.tools import get_input, init_dist

from petrel_client.client import Client
client = Client()

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

FILEPATH = 'data.json'
CONV_TEMPLATE = {
    'llava-v1.5-13b': 'vicuna_v1',
    'llava-v1.6-vicuna-13b': 'vicuna_v1',
    'llava-v1.6-34b': 'chatml_direct',
    'llava-next-110b': 'qwen_1_5',
    'VILA1.0-13b-llava': 'vicuna_v1',
}
NUM_HIDDEN_LAYERS = {
    'llava-v1.5-13b': 40,
    'llava-v1.6-vicuna-13b': 40,
    'llava-v1.6-34b': 60,
    'llava-next-110b': 80,
    'VILA1.0-13b-llava': 40,
}

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
    model_name = os.path.basename(args.model_path)
    num_layers = NUM_HIDDEN_LAYERS[model_name]
    num_layers_per_gpu = num_layers // num_gpus

    visible_devices = [i for i in range(args.local_rank, num_gpus, args.local_world_size)]
    device_map = {
        f'model.layers.{i}': visible_devices[min(i // num_layers_per_gpu, len(visible_devices) - 1)]
        for i in range(num_layers)
    }
    device_map['model.vision_tower'] = visible_devices[0]
    device_map['vision_tower'] = visible_devices[0]
    device_map['vision_model'] = visible_devices[0]
    device_map['model.mm_projector'] = visible_devices[0]
    device_map['model.norm'] = visible_devices[0]
    device_map['model.image_newline'] = visible_devices[0]
    device_map['model.embed_tokens'] = visible_devices[0]
    device_map['lm_head'] = visible_devices[-1]

    if args.rank == 0:
        for k, v in device_map.items():
            print(k, v)

    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=args.model_path,
        model_base=None,
        model_name=get_model_name_from_path(args.model_path),
        # device=args.local_rank,
        device_map=device_map,
    )
    tokenizer.model_max_length = 256000
    model.config.max_length = 256000
    model.config.tokenizer_model_max_length = 256000

    if model.config.image_aspect_ratio == 'anyres':
        model.config.image_aspect_ratio = 'pad'
        model.config.mm_patch_merge_type = 'flat'

    return model, tokenizer, image_processor

def main(args):
    init_dist(args)

    task = args.task
    model_name = os.path.basename(args.model_path)
    model, tokenizer, image_processor = build_model(args)

    print(
        f"Rank [{args.rank}] "
        f"Begin to eval model {args.model_path} on task {task}, "
        f"conv_template: {CONV_TEMPLATE[model_name]}, "
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

        # 加载图像
        if len(images_list) > 0:
            images = load_images(images_list)
            image_sizes = [image.size for image in images]
            images_tensor = process_images(
                images,
                image_processor,
                model.config
            )
            if isinstance(images_tensor, torch.Tensor):
                images_tensor = images_tensor.to(model.device, dtype=torch.float16)
            else:
                images_tensor = [t.to(model.device, dtype=torch.float16) for t in images_tensor]
        else:
            image_sizes = None
            images_tensor = None

        conv_mode = CONV_TEMPLATE[model_name]
        conv = conv_templates[conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt",
        ).unsqueeze(0).to(args.local_rank)

        try:
            output_ids = model.generate(
                inputs=input_ids, 
                images=images_tensor, 
                image_sizes=image_sizes, 
                do_sample=False,
                use_cache=True,
                num_beams=1,
                max_new_tokens=32,
            )
            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
            oom_cnt = 0
        except torch.cuda.OutOfMemoryError:
            print(f"Rank {args.rank} OutOfMemoryError occurs! {input_ids.shape=}, totoal_tokens={sample['meta']['context_length']}")
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
            'total_tokens': sample['meta']['context_length'],
            'position': sample['meta']['placed_depth'],
        }) + "\n")
        ans_file.flush()

    print(f"Rank {args.rank} Finish")
    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for LLaVA models")
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
