import pandas as pd
import os
from sklearn.model_selection import train_test_split

LABELS = [
    "Atelectasis","Cardiomegaly","Effusion","Infiltration","Mass",
    "Nodule","Pneumonia","Pneumothorax","Consolidation","Edema",
    "Emphysema","Fibrosis","Pleural_Thickening","Hernia"
]

def prepare_csv(metadata_path, images_dir, output_dir):
    df = pd.read_csv(metadata_path)

    # Vectorized label assignment (faster than iterrows)
    label_df = df["Finding Labels"].str.get_dummies(sep="|")
    for label in LABELS:
        if label not in label_df.columns:
            label_df[label] = 0
    df = pd.concat([df, label_df[LABELS]], axis=1)

    # Add full image path
    df["filename"] = df["Image Index"].apply(lambda x: os.path.join(images_dir, x))

    # Keep only filename + patient ID + 14 labels
    df = df[["filename","Patient ID"] + LABELS]

    # Patient-wise split
    patients = df["Patient ID"].unique()
    train_patients, test_patients = train_test_split(patients, test_size=0.2, random_state=42)
    val_patients, test_patients = train_test_split(test_patients, test_size=0.5, random_state=42)

    train_df = df[df["Patient ID"].isin(train_patients)]
    val_df   = df[df["Patient ID"].isin(val_patients)]
    test_df  = df[df["Patient ID"].isin(test_patients)]

    # Save CSVs
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)

    print(f"Saved train/val/test CSVs in {output_dir}")

if __name__ == "__main__":
    prepare_csv(
        metadata_path="data/Data_Entry_2017.csv",
        images_dir="data/images",
        output_dir="data/processed"
    )
