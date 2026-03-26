# ingest.py

import json
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# 🔹 Load embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# 🔹 Persistent ChromaDB client (FIXED)
client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True   # 🔥 REQUIRED FIX
    )
)

# 🔹 Create or get collection
collection = client.get_or_create_collection(name="health_data")

# 🔹 Load JSON data
with open("data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# 🔹 Insert data
for i, item in enumerate(data):
    text = f"""
Topic: {item['topic']}
Subtopic: {item['subtopic']}
Question: {item['question']}
Answer: {item['answer']}
"""

    embedding = model.encode(text).tolist()

    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[text],
        metadatas=[{
            "topic": item["topic"],
            "subtopic": item["subtopic"]
        }]
    )

# 🔹 Debug check
print("✅ Total records in DB:", collection.count())

print("✅ Ingestion complete. Data stored persistently.")