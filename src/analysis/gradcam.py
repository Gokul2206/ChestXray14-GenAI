import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Add project root to path
sys.path.append(r"C:\Users\gokul\chestxray_project")

from src.training.train import build_densenet121, build_efficientnet_b0, build_customcnn
from src.data.chestxray_dataset import ChestXrayDataset, val_test_transform

# -------------------
# Grad-CAM utility
# -------------------
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()
        output = self.model(input_tensor)
        loss = output[0, class_idx]
        loss.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = torch.relu(cam)

        cam = F.interpolate(cam, size=input_tensor.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

# -------------------
# Main loop
# -------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChestXrayDataset(
        r"C:\Users\gokul\chestxray_project\data\processed\test.csv",
        transform=val_test_transform
    )

    os.makedirs("report/figures/gradcam", exist_ok=True)

    # Disease names (ChestXray14 order)
    disease_names = [
        "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass","Nodule",
        "Pneumonia","Pneumothorax","Consolidation","Edema","Emphysema",
        "Fibrosis","Pleural_Thickening","Hernia"
    ]

    models_cfg = {
        "densenet121": (build_densenet121, r"C:\Users\gokul\chestxray_project\checkpoints\build_densenet121_best.pth", lambda m: m.features[-4]),
        "efficientnet_b0": (build_efficientnet_b0, r"C:\Users\gokul\chestxray_project\checkpoints\build_efficientnet_b0_best.pth", lambda m: m.features[-2][0]),
        "customcnn": (build_customcnn, r"C:\Users\gokul\chestxray_project\checkpoints\build_customcnn_best.pth", lambda m: m[3])
    }

    for model_name, (builder, ckpt_path, target_fn) in models_cfg.items():
        if not os.path.exists(ckpt_path):
            print(f"Checkpoint not found for {model_name}, skipping.")
            continue

        model = builder().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.eval()

        target_layer = target_fn(model)
        gradcam = GradCAM(model, target_layer)

        # For each disease, find one positive test image
        for class_idx, disease_name in enumerate(disease_names):
            found = False
            for idx in range(len(dataset)):
                image, label = dataset[idx]
                if label[class_idx] == 1:  # positive case
                    input_tensor = image.unsqueeze(0).to(device)
                    cam = gradcam.generate(input_tensor, class_idx)
                    img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)

                    plt.figure(figsize=(6,6))
                    plt.imshow(img_np, alpha=0.8)
                    plt.imshow(cam, cmap="jet", alpha=0.4)
                    plt.axis("off")

                    out_path = f"report/figures/gradcam/{model_name}_{disease_name}.png"
                    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
                    plt.close()

                    found = True
                    break  # stop after first positive example

            if not found:
                print(f"⚠️ No positive example found for {disease_name} in test set.")

        print(f"✅ Saved 14 Grad-CAM overlays for {model_name}")

if __name__ == "__main__":
    main()
