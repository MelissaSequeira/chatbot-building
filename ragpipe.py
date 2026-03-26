# ragpipe.py

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
import ollama
import sqlite3

# ==============================
# 🔹 LOAD EMBEDDING MODEL
# ==============================
model = SentenceTransformer("all-MiniLM-L6-v2")

# ==============================
# 🔹 LOAD CHROMA DB
# ==============================
client = chromadb.Client(
    Settings(
        persist_directory="chroma_db",
        is_persistent=True
    )
)

collection = client.get_collection("health_data")
print("✅ Loaded records:", collection.count())

# ==============================
# 🔹 SQLITE (CHAT HISTORY)
# ==============================
conn = sqlite3.connect("chatbot.db")
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_input TEXT,
    bot_response TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")
conn.commit()

def save_chat(user, bot):
    cursor.execute(
        "INSERT INTO chat_history (user_input, bot_response) VALUES (?, ?)",
        (user, bot)
    )
    conn.commit()

# ==============================
# 🔥 DOMAIN KEYWORDS
# ==============================
health_keywords = [
    "period", "menstruation", "pcos", "hormone",
    "pregnancy", "ovulation", "uterus", "cycle",
    "cramps", "bleeding", "periods", "female",
    "women", "diet", "food"
]

# ==============================
# 🔹 CHAT LOOP
# ==============================
print("\n🤖 Chatbot Ready (Improved RAG)")
print("Type 'exit' to quit\n")

while True:
    query = input("You: ")

    if query.lower() == "exit":
        break

    # 🔥 STEP 1: DOMAIN FILTER
    if not any(word in query.lower() for word in health_keywords):
        print("Bot: ❌ I can only answer women's health-related questions.\n")
        continue

    # 🔹 Embed query
    query_embedding = model.encode(query).tolist()

    # 🔹 Retrieve context (INCREASED RESULTS)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5
    )

    docs = results["documents"][0]

    # 🔍 DEBUG (optional - can remove later)
    print("\n🔍 Retrieved Docs:\n", docs, "\n")

    # 🔹 Build context
    if docs:
        context = "\n".join(docs)
    else:
        context = ""

    # ==============================
    # 🔥 SMART PROMPT (BALANCED)
    # ==============================
    prompt = f"""
You are a women's health assistant.

Rules:
- Prefer using the provided context.
- If context is limited, you may use general knowledge related to women's health.
- Do NOT answer unrelated questions.
- Keep answers simple and helpful.

Context:
{context}

Question:
{query}
"""

    # 🔹 Generate response (Ollama - phi3)
    response = ollama.chat(
        model="phi3",
        messages=[
            {"role": "system", "content": "You are a helpful women's health assistant."},
            {"role": "user", "content": prompt}
        ]
    )

    answer = response["message"]["content"]

    print("Bot:", answer, "\n")

    # 🔹 Save chat
    save_chat(query, answer)

conn.close()