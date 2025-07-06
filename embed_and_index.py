import json
import faiss
import numpy as np
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

with open("transcript_chunks.jsonl", "r", encoding="utf-8") as f:
    chunks = [json.loads(l) for l in f]

texts = [c["text"] for c in chunks]
vectors = []

print(f"🔁 Embedding {len(texts)} segments...")
for i, text in enumerate(texts):
    res = openai.Embedding.create(
        model="text-embedding-3-large",
        input=text
    )
    vectors.append(res["data"][0]["embedding"])
    if i % 50 == 0:
        print(f"  ...{i} embedded")

print("🧠 Building FAISS index...")
index = faiss.IndexFlatL2(len(vectors[0]))
index.add(np.array(vectors).astype("float32"))

faiss.write_index(index, "faiss_index.index")
with open("chunk_metadata.json", "w", encoding="utf-8") as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

print("✅ Done: faiss_index.index + chunk_metadata.json generated")

