"""Extraire et découper les jeux de données."""

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


REQUIRED_COLUMNS = ["rating", "clean_text", "label"]


def extract_fields_from_jsonl(input_path, output_path=None):
    """Extrait les colonnes utiles d'un JSONL vers un DataFrame."""
    input_path = Path(input_path)
    rows = []
    with input_path.open("r", encoding="utf-8") as infile:
        for line in infile:
            record = json.loads(line.strip())
            rows.append(
                {
                    "rating": record.get("rating"),
                    "clean_text": record.get("clean_text", ""),
                    "label": record.get("label", ""),
                }
            )

    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False, encoding="utf-8")
    return df


def validate_dataset(df):
    """Vérifie que les colonnes importantes sont bien présentes."""
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def split_dataset(
    input_csv,
    output_dir,
    prefix,
    train_size=0.8,
    dev_size=0.1,
    test_size=0.1,
    random_state=42,
):
    """Crée les fichiers train/dev/test avec une répartition stratifiée."""
    if round(train_size + dev_size + test_size, 5) != 1.0:
        raise ValueError("La somme de train_size, dev_size et test_size doit être égale à 1.0")

    df = pd.read_csv(input_csv)
    validate_dataset(df)

    # On fait d'abord train / reste, puis on recoupe le reste en dev / test.
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_size),
        stratify=df["label"],
        random_state=random_state,
    )
    dev_fraction_of_temp = dev_size / (dev_size + test_size)
    dev_df, test_df = train_test_split(
        temp_df,
        train_size=dev_fraction_of_temp,
        stratify=temp_df["label"],
        random_state=random_state,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "train": output_dir / f"{prefix}_train.csv",
        "dev": output_dir / f"{prefix}_dev.csv",
        "test": output_dir / f"{prefix}_test.csv",
    }
    train_df.to_csv(paths["train"], index=False)
    dev_df.to_csv(paths["dev"], index=False)
    test_df.to_csv(paths["test"], index=False)
    return paths
