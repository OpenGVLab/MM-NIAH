import io
import os
import sys
import json
import logging
import base64
import gradio as gr

from PIL import Image

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    datefmt='%m/%d/%Y %H:%M:%S',
    handlers=[logging.StreamHandler(sys.stdout)],
)

FILEPATH = 'shells/data/mm_niah.json'
IMAGE_PLACEHOLDER = '<image>'

class InterleavedDataset:
    def __init__(self, meta):
        self.image_path = meta['root']
        self.data_path = meta['annotation']

        with open(self.data_path) as file:
            self.lines = file.readlines()

    def __getitem__(self, index):
        item = self.lines[index]
        item = json.loads(item)
        item['image_dir'] = self.image_path
        return item.copy()

    def __len__(self):
        return len(self.lines)

def load_image(image_file):
    image = Image.open(image_file).convert('RGB')
    return image

def image_to_mdstring(image):
    if isinstance(image, str):
        image = load_image(image)

    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"![image](data:image/jpeg;base64,{img_str})"

def process_item(item):
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

    images_list = [os.path.join(image_dir, i) for i in images_list]
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
    for needle in needles:
        if isinstance(needle, int):
            continue

        if needle in context:  # text needles
            context = context.replace(needle, f' `{needle}` ')
        else:  # image needles
            pass

    # choices
    if choices:
        for c_idx, c in enumerate(choices):
            question = f"{question}\n\n{chr(c_idx + ord('A'))}. {c}"

    # choices_image
    if choices_image:
        for c_idx, c in enumerate(choices_image):
            c = image_to_mdstring(os.path.join(image_dir, c))
            question = f"{question}\n\n{chr(c_idx + ord('A'))}. {c}"

    if isinstance(answer, list):
        answer = json.dumps(answer)

    key_list = ['needles', 'placed_depth', 'context_length', 'num_images']
    md_str = [
        '## Meta Info',
        *[f'{k}={meta[k]}' for k in key_list],
        f"num_images={len(images_list)=}",
        '## Context', context,
        '## Question', question,
        '## Answer', answer,
    ]
    md_str = '\n\n'.join(md_str)

    return md_str.replace('<', '\\<').replace('>', '\\>')

def gradio_app_vis_incontext_trainset(_filepath):
    with open(_filepath) as file:
        _filepath = json.load(file)

    def load_and_collate_annotations(ann_filename):
        dataset = InterleavedDataset(_filepath[ann_filename])
        return dataset

    def when_btn_next_click(user_state, ann_filename, ann_id, md_annotation):
        ann_id = int(ann_id) + 1
        item = user_state[ann_filename][ann_id]
        md_annotation = process_item(item)
        return ann_filename, ann_id, md_annotation

    def when_btn_reset_click(user_state, ann_filename, ann_id, annotation):
        return when_btn_next_click(user_state, ann_filename, -1, annotation)

    def when_ann_filename_change(user_state, ann_filename, ann_id, annotation):
        obj = user_state.get(ann_filename, None) 
        if obj is None:
            obj = load_and_collate_annotations(ann_filename)
            user_state[ann_filename] = obj

        return when_btn_next_click(user_state, ann_filename, -1, annotation)

    with gr.Blocks() as app:
        ann_filename = gr.Radio(list(_filepath.keys()), value=None)
        with gr.Row():
            ann_id = gr.Number(0)
            btn_next = gr.Button("Next")
            btn_reset = gr.Button("Reset")
        annotation = gr.Markdown()

        user_state = gr.State({})
        all_components = [ann_filename, ann_id, annotation]
        ann_filename.change(when_ann_filename_change, [user_state] + all_components, all_components)
        btn_reset.click(when_btn_reset_click, [user_state] + all_components, all_components)
        btn_next.click(when_btn_next_click, [user_state] + all_components, all_components)

    server_port = 10010
    for i in range(10010, 10100):
        cmd = f'netstat -aon|grep {i}'
        with os.popen(cmd, 'r') as file:
            if '' == file.read():
                server_port = i
                break
    app.launch(share=True, server_port=server_port)

if __name__ == "__main__":
    gradio_app_vis_incontext_trainset(FILEPATH)
