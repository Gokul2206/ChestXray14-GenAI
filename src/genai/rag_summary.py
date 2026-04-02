import os
import pandas as pd
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- Paths ---
ROOT = Path(__file__).resolve().parents[2]   # chestxray_project/
LOGS = ROOT / "logs"
INDEX_PATH = ROOT / "src" / "genai" / "kb_index"
OUT = ROOT / "report" / "ragsummaries"
OUT.mkdir(parents=True, exist_ok=True)

# --- Load metrics CSVs ---
def load_metrics_csv(file_path, model_name):
    df = pd.read_csv(file_path)
    df["Model"] = model_name
    return df

resnet_df = load_metrics_csv(LOGS / "resnet18_perlabel.csv", "resnet18")
vgg_df = load_metrics_csv(LOGS / "vgg19_perlabel.csv", "vgg19")
customcnn_df = load_metrics_csv(LOGS / "customcnn_perlabel.csv", "customcnn")

metrics_df = pd.concat([resnet_df, vgg_df, customcnn_df], ignore_index=True)

# --- Load FAISS index ---
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local(str(INDEX_PATH), embeddings, allow_dangerous_deserialization=True)

# --- Load Gemma-2B ---
HF_TOKEN = os.getenv("HF_TOKEN") or "your_huggingface_token_here"
model_name = "google/gemma-2b"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_name, token=HF_TOKEN)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=bnb_config,
    token=HF_TOKEN
)

# --- Helper: build summary for one label ---
def llm_summary(label, metrics_df):
    retrieved_docs = vectorstore.similarity_search(label, k=1)
    kb_context = retrieved_docs[0].page_content if retrieved_docs else "No KB context found."

    model_metrics = []
    for model_name in ["resnet18", "vgg19", "customcnn"]:
        m = metrics_df[(metrics_df["Model"] == model_name) & (metrics_df["Label"] == label)]
        if not m.empty:
            row = m.iloc[0]
            model_metrics.append(f"{model_name} AUROC={row['AUROC']:.3f}, PR-AUC={row['PR-AUC']:.3f}")
    metrics_text = "; ".join(model_metrics) if model_metrics else "No metrics available"

    prompt = (
        f"Here are details about {label}:\n"
        f"{kb_context}\n"
        f"Performance metrics: {metrics_text}\n\n"
        f"Write five numbered sentences summarizing these findings clearly. "
        f"Start directly with sentence 1. "
        f"End with: 'This report is assistive only and not a diagnostic document.'"
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=250,
        do_sample=True,
        top_p=0.9,
        temperature=0.5,
        repetition_penalty=1.1
    )

    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if prompt in summary:
        summary = summary.replace(prompt, "").strip()

    return f"### {label}\n{summary}\n"

# --- Build metrics table (example: resnet18 only) ---
resnet_table = resnet_df.to_markdown(index=False)

# --- Assemble Markdown ---
doc = f"""# Chest X-ray Project Report

## Performance Metrics
{resnet_table}

*(Similar tables for vgg19 and customcnn using their CSVs.)*

---

## Learning Curves
![ResNet18 AUROC](../figures/resnet18_auroc_curve.png)  
![ResNet18 F1](../figures/resnet18_f1_curve.png)  
![ResNet18 Precision](../figures/resnet18_precision_curve.png)  
![ResNet18 Recall](../figures/resnet18_recall_curve.png)

![VGG19 AUROC](../figures/vgg19_auroc_curve.png)  
![VGG19 F1](../figures/vgg19_f1_curve.png)  
![VGG19 Precision](../figures/vgg19_precision_curve.png)  
![VGG19 Recall](../figures/vgg19_recall_curve.png)

![CustomCNN AUROC](../figures/customcnn_auroc_curve.png)  
![CustomCNN F1](../figures/customcnn_f1_curve.png)  
![CustomCNN Precision](../figures/customcnn_precision_curve.png)  
![CustomCNN Recall](../figures/customcnn_recall_curve.png)

---

## Grad-CAM Visualizations
Examples from first 5 test images per model:

![ResNet18 Grad-CAM](../figures/gradcam/resnet18_img0_label0.png)  
![VGG19 Grad-CAM](../figures/gradcam/vgg19_img0_label0.png)  
![CustomCNN Grad-CAM](../figures/gradcam/customcnn_img0_label0.png)

---

## Grounded LLM Summaries
"""

# --- Only generate summaries for 3 labels ---
labels = ["Atelectasis", "Cardiomegaly", "Effusion"]  # manually chosen
for label in labels:
    doc += llm_summary(label, metrics_df)

# --- Limitations & Disclaimer ---
doc += """
---

## Limitations
- Labels mined from reports may be noisy and ambiguous.  
- Predictions are probabilistic and require clinical correlation.  
- Small lesions and overlapping pathologies reduce sensitivity.  

## Disclaimer
This report is assistive only and not a diagnostic document.
"""

# --- Save final Markdown ---
out_file = OUT / "final_report.md"
with open(out_file, "w", encoding="utf-8") as f:
    f.write(doc)

print("Final report saved to:", out_file)