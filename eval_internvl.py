import os
import io
import json
import time
import argparse
import torch
import torchvision.transforms as T

from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel, AutoConfig
from torchvision.transforms.functional import InterpolationMode

from utils.tools import get_input, init_dist

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

FILEPATH = 'shells/data/mm_niah.json'
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def current_time():
    return time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))


def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images


def load_image(image_file, dynamic_image_size=True, input_size=448, max_num=6):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    if dynamic_image_size:
        images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    else:
        images = [image]
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values


def build_model(args):
    num_gpus = torch.cuda.device_count()
    visible_devices = [i for i in range(args.local_rank, num_gpus, args.local_world_size)]

    if len(visible_devices) > 1:
        device_map = {}
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)

        num_gpus_for_vit = 1
        num_gpus_for_llm = len(visible_devices) - num_gpus_for_vit

        num_layers = config.llm_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_llm + 1
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu + num_gpus_for_vit, len(visible_devices) - 1)
            device_map[f'language_model.model.layers.{i}'] = visible_devices[device_idx]

        num_layers = config.vision_config.num_hidden_layers
        num_layers_per_gpu = num_layers // num_gpus_for_vit + 1
        for i in range(num_layers):
            device_idx = min(i // num_layers_per_gpu, num_gpus_for_vit - 1)
            device_map[f'vision_model.encoder.layers.{i}'] = visible_devices[device_idx]

        device_map['vision_model.embeddings'] = visible_devices[0]
        device_map['mlp1'] = visible_devices[num_gpus_for_vit - 1]
        # InternLM2
        device_map['language_model.model.tok_embeddings'] = visible_devices[num_gpus_for_vit]
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.output'] = visible_devices[-1]
        # Qwen2
        device_map['language_model.model.embed_tokens'] = visible_devices[num_gpus_for_vit]
        device_map['language_model.model.norm'] = visible_devices[-1]
        device_map['language_model.lm_head'] = visible_devices[-1]

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


def chat(
    model,
    tokenizer,
    pixel_values,
    num_patches_list,
    question,
    generation_config,
    history=None,
    return_history=False,
    IMG_START_TOKEN='<img>',
    IMG_END_TOKEN='</img>',
    IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
):
    assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id
    if tokenizer.convert_tokens_to_ids('<|im_end|>') != 0:
        eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')  # 92542, InternLM2
    else:
        eos_token_id = tokenizer.eos_token_id

    from utils.conversation import get_conv_template

    template = get_conv_template(model.template)
    if history is None:
        history = []
        for num_patches in num_patches_list:
            assert pixel_values is None or '<image>' in question
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches + IMG_END_TOKEN
            question = question.replace('<image>', image_tokens, 1)
    else:
        for (old_question, old_answer) in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    model_inputs = tokenizer(query, return_tensors='pt')
    input_ids = model_inputs['input_ids'].cuda()
    attention_mask = model_inputs['attention_mask'].cuda()
    generation_config['eos_token_id'] = eos_token_id

    if pixel_values is None:
        print(f'dynamic ViT batch size: None, input_ids: {input_ids.shape}')
    else:
        print(f'dynamic ViT batch size: {pixel_values.size(0)}, input_ids: {input_ids.shape}')
        pixel_values = pixel_values.to(torch.bfloat16).cuda()

    generation_output = model.generate(
        pixel_values=pixel_values,
        input_ids=input_ids,
        attention_mask=attention_mask,
        **generation_config
    )
    response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
    response = response.split('<|im_end|>')[0].strip()  # for InternLM2
    history.append((question, response))
    if return_history:
        return response, history
    return response


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

    temp_dir = f"temp_{model_name}_{task}"
    ans_file_name = f'{model_name}_{task}.jsonl'
    ans_file_path = os.path.join(args.outputs_dir, temp_dir, f"{args.rank}_{args.world_size}_{ans_file_name}")

    os.makedirs(os.path.join(args.outputs_dir, temp_dir), exist_ok=True)

    with open(args.question_file, 'r') as file:
        lines = file.readlines()

    skip_idx = set()
    if os.path.exists(ans_file_path):
        with open(ans_file_path) as file:
            ans_lines = file.readlines()

        for ans_line in ans_lines:
            skip_idx.add(json.loads(ans_line)['question_id'])

    ans_file = open(ans_file_path, 'a')
    lines = lines[args.rank::args.world_size]
    lines = [json.loads(line) for line in lines]
    lines = sorted(lines, key=lambda x:x['meta']['context_length'])

    oom_cnt = 0
    print(f'Rank {args.rank} {len(skip_idx)=}')
    for sample in tqdm(lines, desc=f"Processing {ans_file_name}", disable=args.rank!=0):
        if sample['id'] in skip_idx:
            continue

        if oom_cnt >= 20:
            print(f"[Rank {args.rank}] early stops because of successive failures. {oom_cnt=}")
            ans_file.write(json.dumps({
                "question_id": sample['id'],
                "question": question,
                "answer": sample['answer'],
                "response": 'None',
                'context_length':sample['meta']['context_length'],
                'placed_depth':sample['meta']['placed_depth']
            }) + "\n")
            ans_file.flush()
            continue

        context, images_list, question, answer = get_input(sample)
        images_list = [os.path.join(args.image_folder, i) for i in images_list]

        qs = f'{context}\n{question}'

        if len(images_list) > 0:
            pixel_values = []
            num_patches_list = []
            for img in images_list:
                curr_pixel_values = load_image(img, dynamic_image_size=False)
                pixel_values.append(curr_pixel_values)
                num_patches_list.append(len(curr_pixel_values))
            pixel_values = torch.cat(pixel_values)
        else:
            pixel_values = None
            num_patches_list = []

        try:
            outputs = chat(
                model=model,
                tokenizer=tokenizer,
                pixel_values=pixel_values,
                num_patches_list=num_patches_list,
                question=qs,
                generation_config=dict(
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=32 if 'counting' in task else 16,
                ),
            )
            oom_cnt = 0
        except torch.cuda.OutOfMemoryError:
            print(f"[Rank {args.rank}] OutOfMemoryError occurs! totoal_tokens={sample['meta']['context_length']}")
            outputs = 'None'
            oom_cnt += 1
            torch.cuda.empty_cache()

        outputs = outputs.strip()
        print(f"[{current_time()}] [Rank {args.rank}] totoal_tokens={sample['meta']['context_length']}, {outputs=}")

        ans_file.write(json.dumps({
            "question_id": sample['id'],
            "question": question,
            "answer": sample['answer'],
            "response": outputs,
            'context_length': sample['meta']['context_length'],
            'placed_depth': sample['meta']['placed_depth'],
        }) + "\n")
        ans_file.flush()
        skip_idx.add(sample['id'])

    print(f"[{current_time()}] Rank {args.rank} Finish")
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation script for InternVL-1.5")
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
