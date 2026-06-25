import pandas as pd

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

df = pd.read_csv("data/processed/train.csv")

total_samples = len(df)
pos_counts = df[LABELS].sum().values
neg_counts = total_samples - pos_counts

for i, label in enumerate(LABELS):
    print(f"{label}: Positives={pos_counts[i]}, Negatives={neg_counts[i]}, Ratio={pos_counts[i]/total_samples:.4f}")
