import io
import os
import sys
import json
import random
import logging
import base64
import numpy as np
import gradio as gr

from PIL import Image
from collections import defaultdict
from petrel_client.client import Client

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)

FILEPATH = 'data.json'
IMAGE_PLACEHOLDER = '<image>'

STATE = defaultdict(lambda: defaultdict(int))
SAVEPATH = 'outputs_human'
os.makedirs(SAVEPATH, exist_ok=True)

client = Client()
# x_bins = [1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000, 48000, 64000, 80000, 96000, 128000]
x_bins = [1000, 2000, 4000, 8000, 12000, 16000, 24000, 32000, 40000, 48000, 64000]
y_interval = 0.2

class InterleavedDataset:
    def __init__(self, meta):
        self.image_path = meta['root']
        self.data_path = meta['annotation']

        with open(self.data_path) as file:
            lines = file.readlines()

        self.data = [json.loads(line) for line in lines]
        self.data = sorted(self.data, key=lambda x:(x['meta']['context_length'], x['meta']['placed_depth']))

        # for line in lines:
        #     meta = json.loads(line)['meta']
        #     x = meta['context_length']
        #     y = meta['placed_depth']

        #     x_index = np.digitize(x, x_bins)
        #     y_index = int(y / y_interval)

    def __getitem__(self, index):
        item = self.data[index]
        item['image_dir'] = self.image_path
        return item.copy()

    def __len__(self):
        return len(self.data)

def load_image(image_file):
    if 's3:' in image_file:
        data_bytes = client.get(image_file)
        assert data_bytes is not None, f'fail to load {image_file}'
        data_buff = io.BytesIO(data_bytes)
        image = Image.open(data_buff).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

def image_to_mdstring(image):
    if isinstance(image, str):
        image = load_image(image)

    if image.width == 1024 and image.height == 1024:
        image = image.resize(size=(192, 192))

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"![image](data:image/jpeg;base64,{img_str})"

def process_item(item, num_lines=0, total_lines=0):
    image_dir = item['image_dir']
    images_list = item['images_list']
    context = item['context']
    question = item['question']
    answer = item['answer']
    meta = item['meta']

    needles = meta['needles']
    choices = meta['choices']
    choices_image = meta['choices_image_path']

    # context
    num_image_placeholders_in_context = context.count(IMAGE_PLACEHOLDER)
    num_image_placeholders = context.count(IMAGE_PLACEHOLDER) + question.count(IMAGE_PLACEHOLDER)
    assert num_image_placeholders == len(images_list)

    images_list = [
        os.path.join('s3://public-dataset/OBELISC/raw-images', i[len('obelisc/'):]) if i.startswith('obelisc/') else i
        for i in images_list
    ]
    images_list = [
        os.path.join(image_dir, i) if 's3://' not in i else i
        for i in images_list
    ]
    for i in range(num_image_placeholders):
        context = context.replace(IMAGE_PLACEHOLDER, image_to_mdstring(images_list[i]), 1)

    for i in range(num_image_placeholders_in_context, num_image_placeholders):
        question = question.replace(IMAGE_PLACEHOLDER, image_to_mdstring(images_list[i]), 1)

    # answer
    if isinstance(answer, int):
        if choices or choices_image:
            answer = chr(answer + ord('A'))
        else:
            answer = str(answer)

    # needles
    needle_info = ''
    for needle in needles:
        if isinstance(needle, int):
            continue

        if needle in context:  # 文本针
            context = context.replace(needle, f' `{needle}` ')
        else:  # 图像针
            # assert os.path.exists(os.path.join(image_dir, needle)), os.path.join(image_dir, needle)
            needle_info = f'{needle_info}\n{image_to_mdstring(os.path.join(image_dir, needle))}'

    # choices
    if choices:
        for c_idx, c in enumerate(choices):
            question = f"{question}\n\n{chr(c_idx + ord('A'))}. {c}"

    # choices_image
    if choices_image:
        for c_idx, c in enumerate(choices_image):
            c = image_to_mdstring(os.path.join(image_dir, c))
            question = f"{question}\n\n{chr(c_idx + ord('A'))}. {c}"

    key_list = ['needles', 'placed_depth', 'context_length', 'num_images']
    err_info = "Fail to load"

    if isinstance(answer, list):
        answer = json.dumps(answer)

    md_str = [
        f'Annotation Progress: {num_lines}/{total_lines}',
        '## Meta Info',
        *[f'{k}={meta.get(k, err_info)}' for k in key_list],
        f"num_images={len(images_list)=}",
        '## Needle Info', needle_info,
        '## Question', question,
        '## Context', context,
        # '## Answer', answer,
    ]
    md_str = '\n\n'.join(md_str)

    return md_str.replace('<', '\\<').replace('>', '\\>')

def gradio_app_vis_incontext_trainset(_filepath):
    with open(_filepath) as file:
        _filepath = json.load(file)

    keys = list(_filepath.keys())
    for k in keys:
        if not k.endswith('-sampled'):
            _filepath.pop(k)

    def save_react(user_state, ann_filename, ann_id, md_annotation, user_input):
        item = user_state[ann_filename][ann_id]
        save_path = os.path.join(SAVEPATH, f'{ann_filename}.jsonl')
        with open(save_path, 'a') as file:
            file.write(json.dumps({
                "question_id": item['id'],
                "question": item['question'],
                "answer": item['answer'],
                "response": user_input,
                'total_tokens': item['meta']['context_length'],
                'position': item['meta']['placed_depth'],
            }) + '\n')

        x_index = np.digitize(item['meta']['context_length'], x_bins)
        placed_depth = item['meta']['placed_depth']
        y_index = int(sum(placed_depth)/ len(placed_depth) / y_interval)

        print(ann_filename)
        STATE[x_index][y_index] += 1
        for k, v in STATE.items():
            print(k, v)
        print()

    def get_id_without_answer(user_state, ann_filename):
        all_id = set(range(len(user_state[ann_filename])))
        save_path = os.path.join(SAVEPATH, f'{ann_filename}.jsonl')

        if not os.path.exists(save_path):
            # return list(all_id)
            return 0, 0

        with open(save_path) as file:
            lines = file.readlines()

        state_new = defaultdict(lambda: defaultdict(int))
        exist_id = set()
        for line in lines:
            ann = json.loads(line)
            exist_id.add(ann['question_id'])

            x_index = np.digitize(ann['total_tokens'], x_bins)
            placed_depth = ann['position']
            y_index = int(sum(placed_depth)/ len(placed_depth) / y_interval)
            state_new[x_index][y_index] += 1

        global STATE
        STATE = state_new.copy()

        min_cnt = 1000
        ann_id = None
        candidate_id = list(all_id - exist_id)
        random.shuffle(candidate_id)

        for curr_id in candidate_id:
            ann = user_state[ann_filename][curr_id]
            x_index = np.digitize(ann['meta']['context_length'], x_bins)
            placed_depth = ann['meta']['placed_depth']
            y_index = int(sum(placed_depth)/ len(placed_depth) / y_interval)

            if ann_id is None or state_new[x_index][y_index] < min_cnt:
                ann_id = curr_id
                min_cnt = state_new[x_index][y_index]

            if state_new[x_index][y_index] == 0:
                break

        return ann_id, len(lines)

    def load_and_collate_annotations(ann_filename):
        dataset = InterleavedDataset(_filepath[ann_filename])
        return dataset

    def when_btn_next_click(user_state, ann_filename, ann_id, md_annotation, user_input, force_ann_id=False):
        ann_id = int(ann_id)
        if user_input:
            save_react(user_state, ann_filename, ann_id, md_annotation, user_input)
            user_input = None
            user_state['previous_ann_id'] = ann_id

        if not force_ann_id:
            ann_id, num_lines = get_id_without_answer(user_state, ann_filename)
        else:
            _, num_lines = get_id_without_answer(user_state, ann_filename)

        if 'previous_ann_id' not in user_state:
            user_state['previous_ann_id'] = ann_id

        item = user_state[ann_filename][ann_id]
        md_annotation = process_item(item, num_lines=num_lines, total_lines=len(user_state[ann_filename]))
        return ann_filename, ann_id, md_annotation, user_input

    def when_btn_previous_click(user_state, ann_filename, ann_id, annotation, user_input):
        return when_btn_next_click(user_state, ann_filename, user_state['previous_ann_id'], annotation, user_input, force_ann_id=True)

    def when_ann_filename_change(user_state, ann_filename, ann_id, annotation, user_input):
        obj = user_state.get(ann_filename, None) 
        if obj is None:
            obj = load_and_collate_annotations(ann_filename)
            user_state[ann_filename] = obj

        return when_btn_next_click(user_state, ann_filename, 0, annotation, None)

    with gr.Blocks() as app:
        ann_filename = gr.Radio(list(_filepath.keys()), value=None)
        with gr.Row():
            ann_id = gr.Number(0, interactive=False)
            btn_previous = gr.Button("Previous")
            btn_next = gr.Button("Next")
            btn_input = gr.Text()
        annotation = gr.Markdown()

        user_state = gr.State({})
        all_components = [ann_filename, ann_id, annotation, btn_input]
        ann_filename.change(when_ann_filename_change, [user_state] + all_components, all_components)
        btn_previous.click(when_btn_previous_click, [user_state] + all_components, all_components)
        btn_next.click(when_btn_next_click, [user_state] + all_components, all_components)


    server_port = 10010
    for i in range(10010, 10100):
        cmd = f'netstat -aon|grep {i}'
        with os.popen(cmd, 'r') as file:
            if '' == file.read():
                server_port = i
                break
    app.launch(server_name="0.0.0.0", server_port=server_port, share=True)

if __name__ == "__main__":
    gradio_app_vis_incontext_trainset(FILEPATH)
