import os
import io
import re
import json
import argparse
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, CLIPImageProcessor
from transformers import AutoTokenizer

from utils.tools import init_dist


FILEPATH = 'shells/data/mm_niah.json'


def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image


def load_images(image_files):
    out = []
    for image_file in image_files:
        image = load_image(image_file)
        out.append(image)
    return out


def build_model(args):
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True).cuda().eval()

    image_processor = CLIPImageProcessor.from_pretrained(args.model_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False, add_eos_token=True)
    tokenizer.pad_token_id = 0  # set pad_token_id to 0

    return model, tokenizer, image_processor


@torch.no_grad()
def encode(model, tokenizer, image_processor, question, texts, images):
    prefix = 'summarize:'
    question = prefix + question
    texts = [prefix + t for t in texts]

    pixel_values = image_processor(images=images, return_tensors='pt').pixel_values
    pixel_values = pixel_values.to(torch.bfloat16).cuda()
    input_ids = tokenizer(
        [question] + texts,
        return_tensors='pt',
        max_length=80,
        truncation=True,
        padding='max_length',
    ).input_ids.cuda()

    image_features_list = []
    pixel_values_list = torch.split(pixel_values, 8)
    for p in pixel_values_list:
        image_features_list.append(model.encode_image(image=p, mode='InternVL-G'))
    image_features = torch.cat(image_features_list)

    text_features_list = []
    input_ids_list = torch.split(input_ids, 8)
    for i in input_ids_list:
        text_features_list.append(model.encode_text(text=i))

    text_features = torch.cat(text_features_list)

    # normalized features
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    logits_per_text = text_features[:1] @ text_features[1:].t()
    logits_per_image = text_features[:1] @ image_features.t()

    assert logits_per_text.ndim == 2 and logits_per_text.size(0) == 1
    assert logits_per_image.ndim == 2 and logits_per_image.size(0) == 1

    return logits_per_text[0], logits_per_image[0]


def get_rag_context(args, model, tokenizer, image_processor, question, context, images):
    min_num_images = args.min_num_images
    max_context_length = args.max_context_length

    texts = context.split('<image>')
    texts_splitted = []
    for t in texts:
        if t:
            texts_splitted.extend(re.split(r'(?<=[.!?])', t))
        texts_splitted.append('<image>')

    texts_splitted.pop(-1)
    assert ''.join(texts_splitted) == context

    images_list = [i for i in images]
    images_list = [os.path.join(args.image_folder, i) for i in images_list]

    texts_sim, images_sim = encode(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        question=question,
        texts=texts_splitted,
        images=load_images(images_list),
    )

    scores = []
    texts_sim = texts_sim.tolist()
    images_sim = images_sim.tolist()
    for i in range(len(texts_splitted)):
        if texts_splitted[i] == '<image>':
            texts_sim.pop(0)
            scores.append(images_sim.pop(0))
        else:
            scores.append(texts_sim.pop(0))
    assert len(texts_sim) == 0 and len(images_sim) == 0, f"{len(texts_sim)=}, {len(images_sim)=}"

    num_images = 0
    num_tokens = 0
    flag = [False] * len(texts_splitted)
    sorted_idx = np.argsort(scores).tolist()[::-1]

    for i in sorted_idx:
        if num_tokens >= max_context_length and num_images >= min_num_images:
            break

        if num_tokens >= max_context_length and texts_splitted[i] != '<image>':
            continue

        flag[i] = True
        if texts_splitted[i] == '<image>':
            num_images += 1
            num_tokens += 256
        else:
            num_tokens += len(tokenizer(texts_splitted[i]).input_ids) - 1

    img_idx = 0
    new_context = []
    new_images_list = []

    for flag_idx, should_keep in enumerate(flag):
        text = texts_splitted[flag_idx]
        if should_keep:
            new_context.append(text)

        if text == '<image>':
            if should_keep:
                new_images_list.append(images[img_idx])
            img_idx += 1
    assert img_idx == len(images)

    new_context = ''.join(new_context)
    assert len(new_images_list) == new_context.count('<image>')

    return new_context, new_images_list


def main(args):
    init_dist(args)
    model_name = os.path.basename(args.model_path)
    model, tokenizer, image_processor = build_model(args)

    print(
        f"Rank [{args.rank}] "
        f"Begin to run model {args.model_path} to prepare rag, "
        f"devices: {set([p.device for p in model.parameters()])}"
    )

    ans_file_name = f'rag_with_{model_name}_{args.question_file}'
    ans_file_path = os.path.join(args.outputs_dir, ans_file_name)

    os.makedirs(os.path.dirname(ans_file_path), exist_ok=True)

    with open(args.question_file, 'r') as file:
        lines = file.readlines()

    lines = lines[args.rank::args.world_size]
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x:x['meta']['context_length'])

    outputs = []
    for sample in tqdm(lines, desc=f"Processing {ans_file_name}", disable=args.rank!=0):
        sample_with_rag = sample.copy()
        context = sample['context'].replace('</s>', '')
        question = sample['question']
        images_list = sample['images_list'].copy()

        num_images = question.count('<image>')
        if num_images > 0:
            redundant_images_list = images_list[-num_images:]
            images_list = images_list[:-num_images]
        else:
            redundant_images_list = []

        new_context, new_images_list = get_rag_context(
            args=args,
            model=model,
            tokenizer=tokenizer,
            image_processor=image_processor,
            question=question,
            context=context,
            images=images_list,
        )
        new_images_list.extend(redundant_images_list)
        sample_with_rag['context'] = new_context
        sample_with_rag['images_list'] = new_images_list

        outputs.append(json.dumps(sample_with_rag) + "\n")

    print(f"Rank {args.rank} Finish")

    merged_outputs = [None] * args.world_size
    torch.distributed.all_gather_object(merged_outputs, outputs)

    if args.rank == 0:
        with open(ans_file_path, 'w') as ans_file:
            for outputs in merged_outputs:
                for line in outputs:
                    ans_file.write(line)
        print(f"Rank {args.rank} Finish to save outputs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for InternVL-1.5-RAG")
    parser.add_argument('--model-path', type=str, default='OpenGVLab/InternVL-14B-224px')
    parser.add_argument('--task', type=str, default='')
    parser.add_argument('--outputs-dir', type=str, default='')
    parser.add_argument('--num-gpus-per-rank', type=int, default=1)
    parser.add_argument('--max-context-length', type=int, default=8192)
    parser.add_argument('--min-num-images', type=int, default=0)
    args = parser.parse_args()

    with open(FILEPATH) as file:
        meta = json.load(file)

    args.image_folder = meta[args.task]['root']
    args.question_file = meta[args.task]['annotation']
    main(args)
