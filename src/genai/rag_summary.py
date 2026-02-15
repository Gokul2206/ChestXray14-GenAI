import faiss
import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List

# -------------------
# Load curated KB
# -------------------
def load_kb(kb_path="src/genai/knowledge_base.json"):
    with open(kb_path, "r") as f:
        kb = json.load(f)
    return kb

# -------------------
# Build FAISS index
# -------------------
def build_index(kb, model_name="all-MiniLM-L6-v2"):
    embedder = SentenceTransformer(model_name)
    texts, meta = [], []
    for entry in kb:
        text = f"{entry['label']}: {entry['definition']} {entry['caveats']} {entry['limitations']}"
        texts.append(text)
        meta.append(entry)
    embeddings = embedder.encode(texts, convert_to_numpy=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index, embedder, meta

# -------------------
# Retrieval
# -------------------
def retrieve_notes(labels: List[str], kb, index, embedder, meta, top_k=1):
    results = {}
    for label in labels:
        query_emb = embedder.encode([label], convert_to_numpy=True)
        D, I = index.search(query_emb, top_k)
        retrieved = [meta[i] for i in I[0]]
        results[label] = retrieved
    return results

# -------------------
# Generate summary
# -------------------
def generate_summary(predictions: dict, retrieved_notes: dict):
    summary = []
    summary.append("Findings:")
    for label, conf in predictions.items():
        summary.append(f"- Suggestive evidence of {label} (confidence {conf:.2f})")

    summary.append("\nContext:")
    for label, notes in retrieved_notes.items():
        for n in notes:
            summary.append(f"- {label}: {n['definition']} [KB]. Caveats: {n['caveats']}")

    summary.append("\nLimitations:")
    summary.append("Predictions are probabilistic, labels may contain noise, and clinical correlation is required.")

    summary.append("\nDisclaimer:")
    summary.append("This summary is assistive only and not a diagnostic report.")

    return "\n".join(summary)

# -------------------
# Batch generation for multiple cases
# -------------------
if __name__ == "__main__":
    kb = load_kb("src/genai/knowledge_base.json")
    index, embedder, meta = build_index(kb)

    # Define representative cases
    cases = {
        "case1_summary.txt": {"Atelectasis": 0.92},            # Confident case
        "case2_summary.txt": {"Pleural Effusion": 0.55},       # Borderline case
        "case3_summary.txt": {"Pneumonia": 0.33},              # Failure case
    }

    out_dir = "report/rag_summaries"
    os.makedirs(out_dir, exist_ok=True)

    for filename, predictions in cases.items():
        retrieved = retrieve_notes(list(predictions.keys()), kb, index, embedder, meta)
        summary = generate_summary(predictions, retrieved)
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(summary)
        print(f"Saved summary to {out_dir}/{filename}")