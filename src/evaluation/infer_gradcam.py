import os
import torch
import torch.nn as nn
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import cv2

from src.data.chestxray_dataset import ChestXrayDataset, transform

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
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (input_tensor.shape[2], input_tensor.shape[3]))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def overlay_cam(image, cam, alpha=0.5):
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)
    return overlay

# -------------------
# Build models
# -------------------
def build_model(name):
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 14)
        target_layer = model.layer4[-1]
    elif name == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, 14)
        target_layer = model.features[-1]
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
        # For custom CNN, target layer is second conv
        target_layer = model[3]
    else:
        raise ValueError("Unknown model name")
    return model, target_layer

# -------------------
# Main loop
# -------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = ChestXrayDataset("data/processed/test.csv", transform=transform)

    os.makedirs("report/figures/gradcam", exist_ok=True)

    # Loop through models
    for model_name in ["resnet18", "vgg19", "customcnn"]:
        ckpt_path = f"checkpoints/{model_name}/epoch10.pth"
        if not os.path.exists(ckpt_path):
            continue

        model, target_layer = build_model(model_name)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        model.to(device)
        model.eval()

        gradcam = GradCAM(model, target_layer)

        # Take first 5 test images
        for idx in range(5):
            image, label = dataset[idx]
            input_tensor = image.unsqueeze(0).to(device)

            # Run Grad-CAM for first 3 disease labels
            for class_idx in range(3):
                cam = gradcam.generate(input_tensor, class_idx)
                img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                overlay = overlay_cam(img_np, cam)

                out_path = f"report/figures/gradcam/{model_name}_img{idx}_label{class_idx}.png"
                plt.imsave(out_path, overlay)
        print(f"Saved Grad-CAM overlays for {model_name}")

if __name__ == "__main__":
    main()