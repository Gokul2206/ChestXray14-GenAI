import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Paths ---
ROOT = Path.cwd()
KB_PATH = ROOT / "src" / "genai" / "knowledge_base.json"   # JSON KB file
INDEX_PATH = ROOT / "src" / "genai" / "kb_index"           # FAISS index folder

# --- Load KB ---
with open(KB_PATH, "r", encoding="utf-8") as f:
    kb = json.load(f)

# --- Prepare texts ---
texts = [
    f"Label: {entry['label']}\nDefinition: {entry.get('definition','')}\nContext: {entry.get('context','')}\nLimitations: {entry.get('limitations','')}"
    for entry in kb
]

# --- Build embeddings ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# --- Create FAISS index ---
vectorstore = FAISS.from_texts(texts, embedding=embeddings)

# --- Save index ---
vectorstore.save_local(str(INDEX_PATH))

print("FAISS index built and saved to:", INDEX_PATH)