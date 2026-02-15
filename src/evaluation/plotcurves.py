import matplotlib.pyplot as plt
import json
import os

def plot_metrics(log_file, model_name, out_dir="report/figures"):
    os.makedirs(out_dir, exist_ok=True)

    with open(log_file, "r") as f:
        logs = json.load(f)

    epochs = logs["epochs"]

    # AUROC curve
    plt.figure()
    plt.plot(epochs, logs["mean_auroc"], marker="o", label="Mean AUROC")
    plt.xlabel("Epoch")
    plt.ylabel("AUROC")
    plt.title(f"{model_name} Mean AUROC over Epochs")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{model_name}_auroc_curve.png"))

    # F1 curves
    plt.figure()
    plt.plot(epochs, logs["micro_f1"], marker="o", label="Micro F1")
    plt.plot(epochs, logs["macro_f1"], marker="o", label="Macro F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1 Score")
    plt.title(f"{model_name} F1 Scores over Epochs")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{model_name}_f1_curve.png"))

    # Precision curve
    plt.figure()
    plt.plot(epochs, logs["micro_precision"], marker="o", label="Micro Precision")
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title(f"{model_name} Micro Precision over Epochs")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{model_name}_precision_curve.png"))

    # Recall curve
    plt.figure()
    plt.plot(epochs, logs["micro_recall"], marker="o", label="Micro Recall")
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title(f"{model_name} Micro Recall over Epochs")
    plt.legend()
    plt.savefig(os.path.join(out_dir, f"{model_name}_recall_curve.png"))

    print(f"Saved plots for {model_name} in {out_dir}/")

if __name__ == "__main__":
    plot_metrics("logs/resnet18_relogs.json", "resnet18")
    plot_metrics("logs/vgg19_relogs.json", "vgg19")
    plot_metrics("logs/customcnn_relogs.json", "customcnn")