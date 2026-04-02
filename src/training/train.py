import os, random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score

from src.data.chestxray_dataset import ChestXrayDataset, transform

# -------------------
# Reproducibility
# -------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# -------------------
# Model definitions
# -------------------
def build_model(name):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, 14)
    elif name == "vgg19":
        model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 14)
    elif name == "customcnn":
        model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Linear(256, 14)
        )
    else:
        raise ValueError("Unknown model name")
    return model

# -------------------
# Focal Loss
# -------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce_loss)
        if self.alpha is not None:
            focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()

# -------------------
# Training loop
# -------------------
def train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
    print(f"Epoch {epoch} - Train Loss: {total_loss/len(train_loader):.4f}")

# -------------------
# Evaluation + Threshold tuning
# -------------------
def tune_thresholds(y_true, y_probs, step=0.01):
    num_classes = y_true.shape[1]
    thresholds = []
    for c in range(num_classes):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0, 1, step):
            preds = (y_probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds

def evaluate(model, loader, device, label_columns, split="Val", thresholds=None):
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

    print(f"\n{split} set metrics:")
    for i, disease in enumerate(label_columns):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            ap    = average_precision_score(all_labels[:, i], all_outputs[:, i])
            print(f"{disease}: AUROC={auroc:.3f}, PR-AUC={ap:.3f}")
        except ValueError:
            print(f"{disease}: skipped (no positive samples in {split} set)")

    # Thresholded metrics
    if thresholds is None:
        thresholds = [0.5] * all_labels.shape[1]
    preds = np.zeros_like(all_outputs, dtype=int)
    for i in range(all_labels.shape[1]):
        preds[:, i] = (all_outputs[:, i] >= thresholds[i]).astype(int)

    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_prec = precision_score(all_labels, preds, average="micro", zero_division=0)
    micro_rec  = recall_score(all_labels, preds, average="micro", zero_division=0)

    print(f"\n{split} set summary:")
    print(f"Micro F1={micro_f1:.3f}, Macro F1={macro_f1:.3f}")
    print(f"Micro Precision={micro_prec:.3f}, Micro Recall={micro_rec:.3f}")

    return all_labels, all_outputs

# -------------------
# Main
# -------------------
def main(use_focal=False, use_sampler=True):
    # Load datasets
    train_dataset = ChestXrayDataset("data/processed/train.csv", transform=transform)
    val_dataset   = ChestXrayDataset("data/processed/val.csv", transform=transform)
    test_dataset  = ChestXrayDataset("data/processed/test.csv", transform=transform)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Compute pos_weight
    train_df = pd.read_csv("data/processed/train.csv")
    label_columns = train_dataset.label_columns
    pos_counts = train_df[label_columns].sum().values
    neg_counts = len(train_df) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-8), dtype=torch.float).to(device)

    # Balanced sampler
    if use_sampler:
        sample_weights = []
        for _, row in train_df[label_columns].iterrows():
            weight = (row.values * (neg_counts / (pos_counts + 1e-8))).sum()
            sample_weights.append(weight)
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
    else:
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Train all models
    for model_name in ["resnet18", "vgg19", "customcnn"]:
        print(f"\n=== Training {model_name.upper()} ===")
        model = build_model(model_name).to(device)

        criterion = FocalLoss(alpha=pos_weight, gamma=2.0) if use_focal else nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        os.makedirs(f"checkpoints/{model_name}", exist_ok=True)

        for epoch in range(1, 11):
            train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler)
            evaluate(model, val_loader, device, label_columns, split="Val")

            ckpt_path = f"checkpoints/{model_name}/epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Threshold tuning on validation set
        y_true, y_probs = evaluate(model, val_loader, device, label_columns, split="Val")
        tuned_thresholds = tune_thresholds(y_true, y_probs)

        # Final test evaluation with tuned thresholds
        evaluate(model, test_loader, device, label_columns, split="Test", thresholds=tuned_thresholds)

if __name__ == "__main__":
    main(use_focal=False, use_sampler=True)