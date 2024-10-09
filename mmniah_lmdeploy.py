from lmdeploy import pipeline, GenerationConfig, TurbomindEngineConfig
from lmdeploy.vl.constants import IMAGE_TOKEN
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--file-name', type=str, default='')
parser.add_argument('--model-path', type=str, default='')
parser.add_argument('--file-dir', type=str, default='')
parser.add_argument('--image-dir', type=str, default='')
parser.add_argument('--save-dir', type=str, default='')
args = parser.parse_args()

pipe = pipeline(args.model_path,
                # log_level='INFO',
                backend_config=TurbomindEngineConfig(session_len=1280000, tp=8))



from tqdm import tqdm
import json
import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import inspect
import os

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data
res = []
file_name = os.path.join(args.file_dir, args.file_name + ".jsonl")
jsonl = read_jsonl(file_name)
for i in tqdm(range(len(jsonl))):
    data = jsonl[i]
    pixel_values_list = []
    images_list = data["images_list"]
    if data["meta"]["choices_image_path"]:
        images_list += data["meta"]["choices_image_path"]
    for j in range(len(images_list)):
        images_list[j] = os.path.join(args.image_dir, images_list[j])
        # pixel_values_list.append(load_image(img, max_num=12).to(torch.bfloat16).cuda())
    # pixel_values = torch.cat(tuple(pixel_values_list), dim=0)
    # num_patches_list = [pixel_values.size(0) for pixel_values in pixel_values_list]
    question = data["context"] + "\n" + data["question"]
    
    if data["meta"]["choices"]:
        for c_idx, c in enumerate(data["meta"]["choices"]):
            question = f"{question}\n{chr(c_idx + ord('A'))}. {c}"
        question += "\nAnswer with the option's letter from the given choices directly."

    elif data["meta"]["choices_image_path"]:
        for c_idx, c in enumerate(data["meta"]["choices_image_path"]):
            question = f"{question}\n{chr(c_idx + ord('A'))}. <image>"
        question += "\nAnswer with the option's letter from the given choices directly."
        
    else:
        question += '\nAnswer the question using a single word or phrase.'
        
    content=[dict(type='text', text=question.replace("<image>", IMAGE_TOKEN))]
    for j in range(len(images_list)):
        content.append(dict(type='image_url', image_url=dict(max_dynamic_patch=12, url=images_list[j])))
    messages = [dict(role='user', content=content)]
    out = pipe(messages, gen_config=GenerationConfig(top_k=1))
    res.append({
        "id": i,
        "response": out.text,
        "answer": data["answer"],
        "category": data["meta"]["category"]
    })

with open(os.path.join(args.save_dir, args.file_name + ".jsonl"), "w") as file:
    for item in res:
        file.write(json.dumps(item) + "\n")

