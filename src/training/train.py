import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
import numpy as np
import random, os

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
# Evaluation
# -------------------
def evaluate(model, loader, device, label_columns, split="Val", threshold=0.5):
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
    preds = (all_outputs >= threshold).astype(int)
    micro_f1 = f1_score(all_labels, preds, average="micro", zero_division=0)
    macro_f1 = f1_score(all_labels, preds, average="macro", zero_division=0)
    micro_prec = precision_score(all_labels, preds, average="micro", zero_division=0)
    micro_rec  = recall_score(all_labels, preds, average="micro", zero_division=0)

    print(f"\n{split} set summary:")
    print(f"Micro F1={micro_f1:.3f}, Macro F1={macro_f1:.3f}")
    print(f"Micro Precision={micro_prec:.3f}, Micro Recall={micro_rec:.3f}")

# -------------------
# Main
# -------------------
def main():
    # Load datasets
    train_dataset = ChestXrayDataset("data/processed/train.csv", transform=transform)
    val_dataset   = ChestXrayDataset("data/processed/val.csv", transform=transform)
    test_dataset  = ChestXrayDataset("data/processed/test.csv", transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    if device.type == "cuda":
        print("GPU name:", torch.cuda.get_device_name(0))

    # Train all three models
    for model_name in ["resnet18", "vgg19", "customcnn"]:
        print(f"\n=== Training {model_name.upper()} ===")
        model = build_model(model_name).to(device)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

        os.makedirs(f"checkpoints/{model_name}", exist_ok=True)

        for epoch in range(1, 11):  # 10 epochs
            train_one_epoch(epoch, model, train_loader, criterion, optimizer, device, scaler)
            evaluate(model, val_loader, device, train_dataset.label_columns, split="Val")

            ckpt_path = f"checkpoints/{model_name}/epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

        # Final test evaluation
        evaluate(model, test_loader, device, train_dataset.label_columns, split="Test")

if __name__ == "__main__":
    main()