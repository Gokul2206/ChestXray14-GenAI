ChestXray14: CNN Performance & Interpretability with GenAI Summaries
📑 Overview
This project implements a complete workflow for chest X‑ray analysis using deep learning and interpretability techniques.
It includes:
- Data preprocessing of the ChestXray14 dataset.
- Training CNN models (CustomCNN, ResNet18, VGG19).
- Evaluation with metrics and plots.
- Interpretability using Grad‑CAM overlays.
- GenAI/RAG summaries combining predictions with a curated knowledge base.
- Representative cases (confident, borderline, failure).
📂 Folder Structure
project/
├── data/                 # Raw and preprocessed ChestXray14 data
├── logs/                 # CSV metrics for each model
├── report/
│   ├── figures/          # Performance plots
│   │   └── gradcam/      # Grad-CAM overlays
│   ├── rag_summaries/    # Case summaries (Findings + Context)
├── src/
│   ├── data/             # Preprocessing scripts
│   ├── evaluation/       # Metrics + plots generation
│   └── genai/
│       ├── rag_summary.py
│       └── knowledge_base.json
└── notebooks/
└── project_report.ipynb
⚙️ Requirements
- Python 3.9+
- Libraries:
- torch, torchvision
- sentence-transformers
- faiss
- pandas, matplotlib
- IPython
- Jupyter Notebook or JupyterLab
- Check requirements.txt file for all required dependencies and install
Data Access
The ChestXray14 dataset (~40GB) is not stored in this repository.
Please download it from the NIH Open Data Portal and place it under data/raw/.
🚀 Workflow
Step 1: Data Preprocessing (src/data/)
- Load ChestXray14 dataset.
- Resize images, normalize pixel values.
- Handle missing labels and multi‑label format.
- Split into train/validation/test sets.
- Save preprocessed data in data/.
Step 2: Model Training
- Train CustomCNN, ResNet18, VGG19 on preprocessed data.
- Save evaluation metrics (AUROC, PR‑AUC, F1, Precision, Recall) as CSV files in logs/.
- Checkpoints are saved per model:
- ResNet18 → checkpoints/resnet18/epochX.pth
- VGG19 → checkpoints/vgg19/epochX.pth
- CustomCNN → checkpoints/customcnn/epochX.pth
Step 3: Evaluation (src/evaluation/)
- Load metrics from logs/.
- Generate plots (AUROC, F1, Precision, Recall curves).
- Save them as .png in report/figures/.
Step 4: Interpretability
- Apply Grad‑CAM to predictions.
- Save overlays in report/figures/gradcam/.
Step 5: GenAI/RAG Summaries (src/genai/)
- Run:
python src/genai/rag_summary.py
- Produces multiple case summaries (case1_summary.txt, case2_summary.txt, case3_summary.txt) in report/rag_summaries/.
- Each summary includes:
- Findings → from model predictions (confidence scores).
- Context → from KB definitions/caveats retrieved via FAISS.
- Limitations & Disclaimer → assistive only, not diagnostic.
Step 6: Reporting Notebook
- Open:
jupyter notebook notebooks/project_report.ipynb
- Run all cells to display:
- Metrics tables.
- Performance plots.
- Case summaries with Findings + Context.
- Grad‑CAM overlays for each case.
📊 Outputs
- Metrics tables: AUROC, PR‑AUC, F1, Precision, Recall.
- Performance plots: AUROC, F1, Precision, Recall curves.
- Interpretability section:
- Case summaries (confident, borderline, failure).
- Grad‑CAM overlays.
- Discussion: Strengths, weaknesses, caveats, disclaimers.
⚠️ Disclaimer
- Predictions are probabilistic.
- Labels may contain noise.
- Clinical correlation is required.
- Summaries are assistive only, not diagnostic reports.