import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import torchvision.transforms as T

# -------------------
# Transforms
# -------------------

# Training: augmentations for robustness
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(p=0.5),          # safe for chest X-rays
    T.RandomRotation(degrees=10),           # small rotations
    T.ColorJitter(brightness=0.1, contrast=0.1),  # scanner variability
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# Validation/Test: clean, no augmentations
val_test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

# -------------------
# Dataset Class
# -------------------
class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

        # All label columns (14 diseases)
        self.label_columns = [
            "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
            "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
            "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data.iloc[idx]["filename"]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        labels = self.data.iloc[idx][self.label_columns].values.astype("float32")
        labels = torch.tensor(labels, dtype=torch.float)

        return image, labels
