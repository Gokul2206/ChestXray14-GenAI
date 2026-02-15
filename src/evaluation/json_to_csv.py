import json
import pandas as pd
import os

def json_to_csv(json_file, csv_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    # Convert dict of dicts → DataFrame
    df = pd.DataFrame.from_dict(data, orient="index")
    df.index.name = "Label"

    os.makedirs(os.path.dirname(csv_file), exist_ok=True)
    df.to_csv(csv_file)
    print(f"Saved CSV: {csv_file}")

def main():
    models = ["resnet18", "vgg19", "customcnn"]
    for model in models:
        json_file = f"logs/{model}_perlabel.json"
        csv_file = f"logs/{model}_perlabel.csv"
        if os.path.exists(json_file):
            json_to_csv(json_file, csv_file)

if __name__ == "__main__":
    main()