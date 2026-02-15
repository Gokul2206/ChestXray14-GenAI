import os
import torch
import json
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from src.data.chestxray_dataset import ChestXrayDataset, transform
from src.training.train import build_model

def evaluate_per_label(model, loader, device, label_columns, threshold=0.5):
    model.eval()
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_labels.append(labels.cpu())
            all_outputs.append(outputs.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()

    results = {}
    for i, label in enumerate(label_columns):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            auroc = None
        try:
            prauc = average_precision_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            prauc = None

        preds = (all_outputs[:, i] >= threshold).astype(int)
        f1 = f1_score(all_labels[:, i], preds, zero_division=0)
        prec = precision_score(all_labels[:, i], preds, zero_division=0)
        rec = recall_score(all_labels[:, i], preds, zero_division=0)

        results[label] = {
            "AUROC": auroc,
            "PR-AUC": prauc,
            "F1": f1,
            "Precision": prec,
            "Recall": rec
        }

    return results

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_dataset = ChestXrayDataset("data/processed/val.csv", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    label_columns = val_dataset.label_columns

    for model_name in ["resnet18", "vgg19", "customcnn"]:
        ckpt_path = f"checkpoints/{model_name}/epoch10.pth"
        if not os.path.exists(ckpt_path):
            continue

        model = build_model(model_name).to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

        results = evaluate_per_label(model, val_loader, device, label_columns)

        os.makedirs("logs", exist_ok=True)
        out_path = f"logs/{model_name}_perlabel.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        print(f"Saved per-label metrics: {out_path}")

if __name__ == "__main__":
    main()