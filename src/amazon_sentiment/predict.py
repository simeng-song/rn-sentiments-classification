"""Fonctions de prédiction pour les modèles déjà entraînés"""

from pathlib import Path

import pandas as pd
import torch

from .models import load_bert_artifact, load_logreg_artifact
from .preprocessing import clean_text


# prédit les labels avec la baseline sauvegardée
def predict_with_logreg(artifact_path, texts):
    artifact = load_logreg_artifact(artifact_path)
    # on nettoie les textes ici aussi pour rester cohérents avec l'entraînement
    cleaned_texts = [clean_text(text) for text in texts]
    features = artifact["vectorizer"].transform(cleaned_texts)
    predictions = artifact["model"].predict(features)
    return artifact["label_encoder"].inverse_transform(predictions).tolist()


# prédit les labels avec le modèle distilbert sauvegardé
def predict_with_bert(model_dir, texts):
    model, tokenizer, inverse_mapping = load_bert_artifact(model_dir)
    encoded = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt",
    )
    with torch.no_grad():
        logits = model(**encoded).logits
    predictions = torch.argmax(logits, dim=1).tolist()
    return [inverse_mapping[index] for index in predictions]


# choisit automatiquement le bon type de modèle pour la prédiction
def predict_texts(model_type, model_path, texts):
    if model_type == "logreg":
        return predict_with_logreg(model_path, texts)
    if model_type == "bert":
        return predict_with_bert(model_path, texts)
    raise ValueError(f"Type de modèle non pris en charge : {model_type}")


# lance des prédictions sur un csv et sauve le résultat
def predict_csv(
    model_type,
    model_path,
    input_csv,
    output_csv,
    text_column="text",
):
    df = pd.read_csv(input_csv)
    if text_column not in df.columns:
        raise ValueError(f"La colonne '{text_column}' est introuvable dans {input_csv}")

    df["predicted_label"] = predict_texts(model_type, model_path, df[text_column].fillna("").tolist())
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return output_csv
