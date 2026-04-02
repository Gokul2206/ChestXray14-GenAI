import json
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

ROOT = Path.cwd()
INDEX_PATH = ROOT / "src" / "genai" / "kb_index"

# Load embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load FAISS index
vectorstore = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

# Test query
query = "Pneumonia"
docs = vectorstore.similarity_search(query, k=1)
print("Retrieved snippet:", docs[0].page_content)