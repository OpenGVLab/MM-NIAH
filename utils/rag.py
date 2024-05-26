import re
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def encode_text(texts):
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True)
    outputs = model.get_text_features(**inputs)
    return outputs

def encode_images(images):
    pil_images = [Image.open(image_path) for image_path in images]
    inputs = processor(images=pil_images, return_tensors="pt")
    outputs = model.get_image_features(**inputs)
    return outputs

def split_text(text, max_length=200):
    sentences = re.split(r'(?<=[。！？])', text)
    chunks = []
    current_chunk = ""
    current_length = 0
    for sentence in sentences:
        sentence_length = len(processor.tokenizer.encode(sentence))
        if current_length + sentence_length > max_length:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_length = sentence_length
        else:
            current_chunk += sentence
            current_length += sentence_length
    if current_chunk:
        chunks.append(current_chunk.strip())
    return chunks

def rag(text, images, query, n):
    split_texts = re.split(r'(<image>)', text)
    text_chunks = []
    img_indices = []
    for i, chunk in enumerate(split_texts):
        if chunk == "<image>":
            img_indices.append(len(text_chunks))
            text_chunks.append(chunk)
        else:
            split_chunks = split_text(chunk)
            text_chunks.extend(split_chunks)

    text_features = encode_text([chunk for chunk in text_chunks if chunk != "<image>"])
    image_features = encode_images(images)

    query_features = encode_text([query])

    text_similarities = cosine_similarity(query_features.detach().numpy(), text_features.detach().numpy()).flatten().tolist()
    image_similarities = cosine_similarity(query_features.detach().numpy(), image_features.detach().numpy()).flatten().tolist()

    combined_similarities = []
    text_idx = 0
    image_idx = 0
    for chunk in text_chunks:
        if chunk == "<image>":
            combined_similarities.append((image_similarities[image_idx], chunk, image_idx, 'image'))
            image_idx += 1
        else:
            combined_similarities.append((text_similarities[text_idx], chunk, text_idx, 'text'))
            text_idx += 1
    combined_similarities.sort(key=lambda x: x[0], reverse=True)

    selected_text = []
    selected_images = []
    total_length = 0
    selected_image_indices = set()
    selected_text_indices = set()

    for sim, chunk, idx, chunk_type in combined_similarities:
        if chunk_type == 'image':
            if total_length + 576 <= n:
                selected_image_indices.add(idx)
                total_length += 576
                # print('after add image', total_length)
        else:
            chunk_length = len(processor.tokenizer.encode(chunk))
            if total_length + chunk_length <= n:
                selected_text_indices.add(idx)
                total_length += chunk_length
                # print('after add text', total_length)
    # print(total_length)

    for i, chunk in enumerate(text_chunks):
        if chunk == "<image>":
            if img_indices.index(i) in selected_image_indices:
                selected_text.append(chunk)
                selected_images.append(images[img_indices.index(i)])
        else:
            if i in selected_text_indices:
                selected_text.append(chunk)

    final_text = "".join(selected_text)
    return final_text, selected_images
