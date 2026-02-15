import os
import torch
import json
from torch.utils.data import DataLoader
from src.data.chestxray_dataset import ChestXrayDataset, transform
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import numpy as np

def evaluate_checkpoint(model, loader, device, label_columns, threshold=0.5):
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

    # Mean AUROC across labels
    aurocs = []
    for i in range(len(label_columns)):
        try:
            aurocs.append(roc_auc_score(all_labels[:, i], all_outputs[:, i]))
        except ValueError:
            pass
    mean_auroc = np.mean(aurocs)

    # Thresholded metrics
    preds = (all_outputs >= threshold).astype(int)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_prec = precision_score(all_labels, preds, average="micro", zero_division=0)
    micro_rec  = recall_score(all_labels, preds, average="micro", zero_division=0)

    return {
        "mean_auroc": float(mean_auroc),
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec)
    }

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    val_dataset = ChestXrayDataset("data/processed/val.csv", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    label_columns = val_dataset.label_columns

    for model_name in ["resnet18", "vgg19", "customcnn"]:
        print(f"\n=== Re‑evaluating {model_name.upper()} ===")
        logs = {"epochs": [], "mean_auroc": [], "micro_f1": [], "macro_f1": [],
                "micro_precision": [], "micro_recall": []}

        # Reload each checkpoint
        for epoch in range(1, 11):
            ckpt_path = f"checkpoints/{model_name}/epoch{epoch}.pth"
            if not os.path.exists(ckpt_path):
                continue

            # Build model same way as in train.py
            from src.training.train import build_model
            model = build_model(model_name).to(device)
            model.load_state_dict(torch.load(ckpt_path, map_location=device))

            metrics = evaluate_checkpoint(model, val_loader, device, label_columns)
            logs["epochs"].append(epoch)
            for k in metrics:
                logs[k].append(metrics[k])

        # Save logs to JSON
        os.makedirs("logs", exist_ok=True)
        out_path = f"logs/{model_name}_relogs.json"
        with open(out_path, "w") as f:
            json.dump(logs, f, indent=2)
        print(f"Saved metrics log: {out_path}")

if __name__ == "__main__":
    main()