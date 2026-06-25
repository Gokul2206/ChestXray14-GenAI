import os, random, numpy as np, pandas as pd, torch, json
import torch.nn as nn, torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, precision_score, recall_score
from src.data.chestxray_dataset import ChestXrayDataset, train_transform, val_test_transform

# -------------------
# Reproducibility
# -------------------
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
set_seed(42)

# -------------------
# Backbones
# -------------------
def build_densenet121():
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(model.classifier.in_features, 14))
    return model

def build_efficientnet_b0():
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4),
        nn.Linear(in_features, 14))
    return model

def build_customcnn():
    return nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
        nn.Flatten(),
        nn.Linear(128*28*28, 256), nn.ReLU(), nn.Dropout(0.5),
        nn.Linear(256, 14)
    )

# -------------------
# Threshold tuning
# -------------------
def tune_thresholds(y_true, y_probs, step=0.01):
    thresholds = []
    for c in range(y_true.shape[1]):
        best_f1, best_t = 0, 0.5
        for t in np.arange(0, 1, step):
            preds = (y_probs[:, c] >= t).astype(int)
            f1 = f1_score(y_true[:, c], preds, average="binary", zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds.append(best_t)
    return thresholds

# -------------------
# Evaluation
# -------------------
def evaluate(model, loader, device, label_columns, split="Val", thresholds=None):
    model.eval()
    all_labels, all_outputs = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            all_labels.append(labels.cpu()); all_outputs.append(outputs.cpu())
    all_labels = torch.cat(all_labels).numpy()
    all_outputs = torch.cat(all_outputs).numpy()

    if thresholds is None:
        thresholds = [0.5] * all_labels.shape[1]
    preds = np.zeros_like(all_outputs, dtype=int)
    for i in range(all_labels.shape[1]):
        preds[:, i] = (all_outputs[:, i] >= thresholds[i]).astype(int)

    results = {}
    for i, disease in enumerate(label_columns):
        try:
            auroc = roc_auc_score(all_labels[:, i], all_outputs[:, i])
            prauc = average_precision_score(all_labels[:, i], all_outputs[:, i])
        except ValueError:
            auroc, prauc = None, None
        results[disease] = {
            "AUROC": auroc, "PR-AUC": prauc,
            "F1": f1_score(all_labels[:, i], preds[:, i], zero_division=0),
            "Precision": precision_score(all_labels[:, i], preds[:, i], zero_division=0),
            "Recall": recall_score(all_labels[:, i], preds[:, i], zero_division=0)
        }
    return results, all_labels, all_outputs

# -------------------
# Main Training Loop with Robust Resume
# -------------------
def main(model_builder=build_densenet121):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Epoch defaults
    if model_builder == build_densenet121: epochs = 20
    elif model_builder == build_efficientnet_b0: epochs = 10
    elif model_builder == build_customcnn: epochs = 8
    else: epochs = 10
    patience = 5

    # Dataset setup
    train_dataset = ChestXrayDataset("data/processed/train.csv", transform=train_transform)
    val_dataset   = ChestXrayDataset("data/processed/val.csv", transform=val_test_transform)
    test_dataset  = ChestXrayDataset("data/processed/test.csv", transform=val_test_transform)
    label_columns = train_dataset.label_columns

    # Class imbalance
    train_df = pd.read_csv("data/processed/train.csv")
    pos_counts = train_df[label_columns].sum().values
    neg_counts = len(train_df) - pos_counts
    pos_weight = torch.tensor(neg_counts / (pos_counts + 1e-8), dtype=torch.float).to(device)

    sample_weights = []
    for _, row in train_df[label_columns].iterrows():
        weight = (row.values * (neg_counts / (pos_counts + 1e-8))).sum()
        sample_weights.append(weight)
    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler, num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4, pin_memory=True)

    # Model + Training
    model = model_builder().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    metrics_file = f"logs/{model_builder.__name__}_metrics.csv"
    perlabel_file = f"logs/{model_builder.__name__}_perlabel.csv"
    checkpoint_path = f"checkpoints/{model_builder.__name__}_checkpoint.pth"
    best_path = f"checkpoints/{model_builder.__name__}_best.pth"

    best_macro_f1 = 0
    best_thresholds = None
    patience_counter = 0
    start_epoch = 1

    # --- Resume if checkpoint exists ---
    if os.path.exists(checkpoint_path):
        print("Resuming from checkpoint...")
        try:
            checkpoint = torch.load(checkpoint_path, weights_only=False)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', optimizer.state_dict()))
                scheduler.load_state_dict(checkpoint.get('scheduler_state_dict', scheduler.state_dict()))
                scaler.load_state_dict(checkpoint.get('scaler_state_dict', scaler.state_dict()))
                best_macro_f1 = checkpoint.get('best_macro_f1', 0)
                best_thresholds = checkpoint.get('thresholds', None)
                start_epoch = checkpoint.get('epoch', 0) + 1
            else:
                print("Checkpoint is weights-only, loading model weights only...")
                model.load_state_dict(checkpoint)
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file)
                    last_epoch = metrics_df['epoch'].max()
                    start_epoch = int(last_epoch) + 1
                    print(f"Detected last epoch {last_epoch}, resuming from {start_epoch}")
                else:
                    start_epoch = 1
        except Exception as e:
            print(f"Failed to load full checkpoint ({e}), loading weights only...")
            model.load_state_dict(torch.load(checkpoint_path))
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
                last_epoch = metrics_df['epoch'].max()
                start_epoch = int(last_epoch) + 1
                print(f"Detected last epoch {last_epoch}, resuming from {start_epoch}")
            else:
                start_epoch = 1

    # --- Fallback: resume from best weights only ---
    elif os.path.exists(best_path):
        print("Resuming from best weights only...")
        model.load_state_dict(torch.load(best_path))
        if os.path.exists(metrics_file):
            metrics_df = pd.read_csv(metrics_file)
            last_epoch = metrics_df['epoch'].max()
            start_epoch = int(last_epoch) + 1
            print(f"Detected last epoch {last_epoch}, resuming from {start_epoch}")
        else:
            start_epoch = 1

    # Training Loop
    for epoch in range(start_epoch, epochs+1):
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
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} - Train Loss: {avg_loss:.4f}")

        # Validation
        results, y_true, y_probs = evaluate(model, val_loader, device, label_columns, split="Val")
        thresholds = tune_thresholds(y_true, y_probs)
        tuned_results, _, _ = evaluate(model, val_loader, device, label_columns, split="Val (Tuned)", thresholds=thresholds)
        tuned_macro_f1 = np.mean([m["F1"] for m in tuned_results.values()])

        scheduler.step(avg_loss)

        # Save per-epoch metrics
        aurocs = [m["AUROC"] for m in results.values() if m["AUROC"] is not None]
        mean_auroc = np.mean(aurocs) if len(aurocs) > 0 else None

        epoch_row = pd.DataFrame([[epoch, mean_auroc,
            f1_score(y_true, (y_probs >= 0.5).astype(int), average="micro", zero_division=0),
            f1_score(y_true, (y_probs >= 0.5).astype(int), average="macro", zero_division=0),
            precision_score(y_true, (y_probs >= 0.5).astype(int), average="micro", zero_division=0),
            recall_score(y_true, (y_probs >= 0.5).astype(int), average="micro", zero_division=0)
        ]], columns=["epoch","mean_auroc","micro_f1","macro_f1","micro_precision","micro_recall"])

        if not os.path.exists(metrics_file):
            epoch_row.to_csv(metrics_file, index=False)
        else:
            epoch_row.to_csv(metrics_file, mode="a", header=False, index=False)

        # Save best model + checkpoint
        if tuned_macro_f1 > best_macro_f1:
            best_macro_f1 = tuned_macro_f1
            best_thresholds = thresholds
            torch.save(model.state_dict(), best_path)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_macro_f1': best_macro_f1,
                'thresholds': best_thresholds
            }, checkpoint_path)
            print(f"Saved new best model at epoch {epoch} with Macro F1={best_macro_f1:.3f}")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience counter = {patience_counter}/{patience}")
            if patience_counter >= patience:
                print("Early stopping triggered.")
                break

    # --- Final Test Evaluation ---
    model.load_state_dict(torch.load(best_path))
    test_results, _, _ = evaluate(model, test_loader, device, label_columns, split="Test", thresholds=best_thresholds)

    print("\nFinal Test Results:")
    perlabel_df = pd.DataFrame.from_dict(test_results, orient="index")
    perlabel_df.index.name = "Label"
    perlabel_df.to_csv(perlabel_file)
    print(f"Saved per-label metrics: {perlabel_file}")

if __name__ == "__main__":
    # main(model_builder=build_densenet121)
    # main(model_builder=build_efficientnet_b0)
    main(model_builder=build_customcnn)
