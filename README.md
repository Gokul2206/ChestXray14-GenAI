```markdown
# NIH ChestX‑ray14: Multi‑label CNNs + RAG‑Grounded LLM Interpretation

## 📑 Overview
This project implements a full pipeline for chest X‑ray analysis using deep learning and GenAI interpretability.  
It combines:
- **Data preprocessing** of NIH ChestX‑ray14.  
- **CNN training** (DenseNet121, EfficientNet‑B0, CustomCNN).  
- **Evaluation** with AUROC, PR‑AUC, F1, Precision, Recall.  
- **Interpretability** using Grad‑CAM overlays.  
- **LLM + RAG summaries** (handled in `project_report.ipynb`) that compare our results against published ChestX‑ray14 benchmarks (Wang et al. 2017 AUROC, CheXNet PR‑AUC).  
- **Final reporting** with a consolidated PDF.

---

## 📂 Folder Structure
```
project/
├── .env                  # API keys
├── requirements.txt      # Python dependencies
├── checkpoints/          # Model weights
│   ├── build_customcnn_best.pth
│   ├── build_densenet121_best.pth
│   └── build_efficientnet_b0_best.pth
├── data/
│   ├── Data_Entry_2017.csv        # NIH metadata
│   ├── benchmarks/
│   │   └── benchmarks_extended.csv # AUROC (Wang 2017) + PR-AUC (CheXNet)
│   ├── images/                    # ChestXray14 images (~112k, excluded from repo)
│   └── PROCESSED/                 # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── logs/                # Metrics per model
│   ├── build_densenet121_perlabel.csv
│   ├── build_efficientnet_b0_perlabel.csv
│   ├── build_customcnn_perlabel.csv
│   └── *_metrics.csv
├── notebooks/
│   └── project_report.ipynb       # End-to-end reporting (LLM + RAG + PDF)
├── report/
│   ├── final_report.pdf
│   └── figures/
│       └── gradcam/               # Grad-CAM overlays
│           ├── customcnn_Atelectasis.png
│           └── ... (more overlays)
└── src/
    ├── data/                      # Preprocessing scripts
    │   ├── chestxray_dataset.py
    │   ├── prepare_nih14.py
    │   └── run_dataset.py
    ├── analysis/                  # Grad-CAM generation
    │   └── gradcam.py
    └── training/                  # Model training
        └── train.py
```

---

## ⚙️ Requirements
- Python 3.9+
- Libraries:  
  `torch`, `torchvision`, `timm`, `pandas`, `numpy`, `scikit-learn`,  
  `albumentations`, `opencv-python`, `pillow`, `pydicom`, `tqdm`,  
  `matplotlib`, `seaborn`, `torchmetrics`, `einops`,  
  `faiss-cpu`, `sentence-transformers`, `transformers`, `sentencepiece`, `accelerate`,  
  `mlflow`, `wandb`, `fastapi`, `uvicorn[standard]`, `python-multipart`, `requests`,  
  `langchain`, `langchain-google-genai`, `python-dotenv`

Install via:
```bash
pip install -r requirements.txt
```

---

## 📥 Dataset Access
The ChestXray14 dataset (~40GB, 112k images) is **not stored in this repo**.  
Download from NIH Box or Kaggle:

- NIH Box: `https://nihcc.app.box.com/v/ChestXray-NIHCC` [(nihcc.app.box.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fnihcc.app.box.com%2Fv%2FChestXray-NIHCC")  
- Kaggle: `https://www.kaggle.com/datasets/nih-chest-xrays/data` [(kaggle.com in Bing)](https://www.bing.com/search?q="https%3A%2F%2Fwww.kaggle.com%2Fdatasets%2Fnih-chest-xrays%2Fdata")  

Place images under:
```
data/images/
```

---

## 🚀 Workflow

### Step 1: Data Preprocessing
```bash
python src/data/prepare_nih14.py
```
- Loads NIH ChestX‑ray14 (`Data_Entry_2017.csv` + images).  
- Resizes, normalizes, handles multi‑label format.  
- Patient‑wise train/val/test splits → saved in `data/PROCESSED/`.

### Step 2: Model Training
```bash
python src/training/train.py --model densenet121
python src/training/train.py --model efficientnet_b0
python src/training/train.py --model customcnn
```
- Trains DenseNet121, EfficientNet‑B0, CustomCNN.  
- Saves checkpoints in `checkpoints/`.  
- Logs metrics in `logs/`.

### Step 3: Evaluation & Interpretability
```bash
python src/analysis/gradcam.py --model densenet121 --label Atelectasis
```
- Generates Grad‑CAM overlays.  
- Saves in `report/figures/gradcam/`.

### Step 4: LLM + RAG Summaries & Final Report
Open the notebook:
```bash
jupyter notebook notebooks/project_report.ipynb
```
- Ingests local logs + external benchmarks (`benchmarks_extended.csv`).  
- Builds FAISS index with HuggingFace embeddings.  
- Uses Gemini LLM to generate citation‑backed summaries.  
- Exports consolidated PDF → `report/final_report.pdf`.

---

## 📊 Outputs
- **Metrics tables**: AUROC, PR‑AUC, F1, Precision, Recall.  
- **Grad‑CAM overlays**: visual interpretability.  
- **LLM Summaries**: grounded, cited text comparing models vs. ChestXray14 benchmarks.  
- **Final Report**: consolidated PDF with metrics, plots, overlays, and summaries.  

---

## ⚠️ Disclaimer
- Predictions are probabilistic and subject to label noise.  
- Clinical correlation is required.  
- Summaries are **assistive only** — not diagnostic reports.
```