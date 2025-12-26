import numpy as np
from pypdf import PdfReader
import faiss
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import requests
from api import get_github_api,get_euri_api

# Load model once
# model_name = "sentence-transformers/all-MiniLM-L6-v2"
# tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = TFAutoModel.from_pretrained(model_name, from_pt=True)

def load_pdf(file):
    text = ""
    reader = PdfReader(file)

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    return text

def create_chunk(text, max_words=100):
    words = text.split()
    chunks = []

    for i in range(0, len(words), max_words):
        chunks.append(" ".join(words[i:i+max_words]))

    return chunks

# def generate_embeddings(text):
#     inputs = tokenizer(
#         text,
#         padding=True,
#         truncation=True,
#         return_tensors="tf"
#     )
#     outputs = model(**inputs)
#     embedding = tf.reduce_mean(outputs.last_hidden_state, axis=1)
#     return embedding.numpy()[0]



import requests
import numpy as np

# Function to get embedding from EURI API
def generate_embeddings(text):
    EURI_API_KEY=get_euri_api()
    url = "https://api.euron.one/api/v1/euri/embeddings"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {EURI_API_KEY}"
    }
    payload = {
        "input": text,
        "model": "text-embedding-3-small"
    }

    response = requests.post(url, headers=headers, json=payload)
    data = response.json()
    
    embedding = np.array(data['data'][0]['embedding'])
    
    return embedding


def create_index(chunks):
    dim = generate_embeddings(chunks[0]).shape[0]
    index = faiss.IndexFlatL2(dim)
    chunk_mapping = []

    for chunk in chunks:
        emb = generate_embeddings(chunk)
        index.add(np.array([emb]).astype("float32"))
        chunk_mapping.append(chunk)

    return index, chunk_mapping

def retrive_top_k(chunk_mapping, index, query, k=3):
    query_emb = generate_embeddings(query)
    _, indices = index.search(
        np.array([query_emb]).astype("float32"), k
    )
    return [chunk_mapping[i] for i in indices[0]]

def build_prompt(chunks, query):
    context = "\n\n".join(chunks)
    return f"""
Use the following context to answer the question.
Context:{context},Question:{query},Answer:
"""

def generate_completion(prompt, model="openai/gpt-4o-mini"):
    api = get_github_api()
    url = "https://models.github.ai/inference/chat/completions"

    headers = {
        "Authorization": f"Bearer {api}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 800
    }

    response = requests.post(url, headers=headers, json=payload)
    return response.json()["choices"][0]["message"]["content"]
