# generate_bert_embeddings.py
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm

# نام مدل از پیش آموزش دیده فارسی
MODEL_NAME = "HooshvareLab/bert-fa-base-uncased"

print("Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
print(f"Using device: {device}")

# خواندن کپشن بنرها
with open('data/raw/banners.txt', 'r', encoding='utf-8') as f:
    captions = [line.strip() for line in f.readlines()]

all_embeddings = []
print("Generating embeddings for all banners...")
with torch.no_grad():
    for text in tqdm(captions):
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
        outputs = model(**inputs)
        # ما از بردار [CLS] به عنوان نمایش کل جمله استفاده می‌کنیم
        cls_embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(cls_embedding)

# ذخیره کردن بردارها
final_embeddings = np.vstack(all_embeddings)
print(f"Embeddings generated with shape: {final_embeddings.shape}") # e.g., (321, 768)
np.save('artifacts/banner_bert_embeddings.npy', final_embeddings)
print("✅ BERT embeddings saved to artifacts/banner_bert_embeddings.npy")