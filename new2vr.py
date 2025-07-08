# Install dependencies first (terminal or notebook)
# pip install sentence-transformers transformers chromadb pdf2image pytesseract Pillow torch

import os #
import torch
import pytesseract #
import subprocess
import numpy as np
from PIL import Image #
from base64 import b64encode
from pdf2image import convert_from_path #
from sentence_transformers import SentenceTransformer #
from transformers import CLIPProcessor, CLIPModel
import chromadb

# --- Step 1: Convert PDF pages to images and extract OCR text ---
pdf_path = './docs/Basic-Electrical-Theory.pdf'
image_folder = 'pdf2_images'
os.makedirs(image_folder, exist_ok=True)

pages = convert_from_path(pdf_path)
image_paths = []
page_texts = []

for i, page in enumerate(pages):
    img_path = os.path.join(image_folder, f'page_{i+1}.png')
    page.save(img_path, 'PNG')
    image_paths.append(img_path)

    text = pytesseract.image_to_string(Image.open(img_path))
    page_texts.append({'image_path': img_path, 'text': text})

# --- Step 2: Load models ---
text_model = SentenceTransformer('all-MiniLM-L6-v2')
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Step 3: Initialize ChromaDB (new style setup) ---
client = chromadb.HttpClient(host="localhost", port=8000)  # Make sure the Chroma server is running
collection = client.get_or_create_collection(name="visual_rag_pages")

# --- Step 4: Image embedding helper ---
def get_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = clip_model.get_image_features(**inputs)
    return outputs[0].cpu().numpy()

# --- Step 5: Index pages with text and store image embeddings separately ---
image_embeddings = {}

for idx, page in enumerate(page_texts):
    text_emb = text_model.encode(page['text'])
    image_emb = get_image_embedding(page['image_path'])

    page_id = f"page_{idx+1}"
    image_embeddings[page_id] = image_emb  # Store separately

    collection.add(
        documents=[page['text']],
        embeddings=[text_emb.tolist()],
        metadatas=[{
            'image_path': page['image_path'],
            'page_number': idx + 1
        }],
        ids=[page_id]
    )

print("✅ PDF indexed with hybrid embeddings (text + image)")

# --- Step 6: Query function using text + reranking with CLIP ---
def hybrid_retrieve(query, top_k=3):
    text_query_emb = text_model.encode(query)
    clip_inputs = clip_processor(text=[query], return_tensors="pt", padding=True)
    with torch.no_grad():
        image_query_emb = clip_model.get_text_features(**clip_inputs).cpu().numpy()

    results = collection.query(query_embeddings=[text_query_emb.tolist()], n_results=top_k)

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    best_score = -1
    best_img_path = None

    for i in range(top_k):
        page_id = results['ids'][0][i]
        meta = results['metadatas'][0][i]
        img_emb = image_embeddings[page_id]
        score = cosine_sim(image_query_emb[0], img_emb)

        if score > best_score:
            best_score = score
            best_img_path = meta['image_path']

    return best_img_path

# --- Step 7: Run vision model on top-ranked image ---
def run_llama_vision(image_path, query):
    with open(image_path, "rb") as img_file:
        encoded_image = b64encode(img_file.read()).decode('utf-8')

    prompt = f"<image>{encoded_image}</image>\n{query}"
    result = subprocess.run(
        ["ollama", "run", "llama3.2-vision"],
        input=prompt.encode(),
        capture_output=True
    )
    print("🧠 Model Response:\n", result.stdout.decode())

# --- Step 8: Test the full pipeline ---
query = "Describe the diagram of a carbon atom"
top_image = hybrid_retrieve(query)
print("🔍 Best matching image:", top_image)
run_llama_vision(top_image, query)
