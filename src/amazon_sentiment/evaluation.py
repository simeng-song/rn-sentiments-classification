"""Evaluation partagées entre les différents modèles"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


# calcule les métriques principales pour un split
def compute_metrics_from_predictions(y_true, y_pred, label_names):
    matrix = confusion_matrix(y_true, y_pred).tolist()
    report = classification_report(y_true, y_pred, target_names=label_names, output_dict=True)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "confusion_matrix": matrix,
        "report": report,
    }


# sauvegarde les métriques au format json
def save_metrics(metrics, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(metrics, outfile, ensure_ascii=False, indent=2)


# recharge un fichier json de métriques
def load_metrics(path):
    with Path(path).open("r", encoding="utf-8") as infile:
        return json.load(infile)


# construit un petit tableau comparatif à partir de plusieurs fichiers
def metrics_summary_frame(metrics_files):
    rows = []
    for metrics_file in metrics_files:
        metrics = load_metrics(metrics_file)
        test_metrics = metrics.get("test", {})
        rows.append(
            {
                "run_name": metrics.get("run_name"),
                "model_type": metrics.get("model_type"),
                "dataset_name": metrics.get("dataset_name"),
                "accuracy": test_metrics.get("accuracy"),
                "f1_macro": test_metrics.get("f1_macro"),
                "metrics_file": str(metrics_file),
            }
        )
    return pd.DataFrame(rows)


# créer un tableau simple pour les prédictions et l'analyse d'erreurs
def prediction_frame(texts, true_labels, pred_labels):
    return pd.DataFrame(
        {
            "text": list(texts),
            "true_label": list(true_labels) if true_labels is not None else [None] * len(texts),
            "predicted_label": list(pred_labels),
        }
    )


# fonction de métriques utilisée pendant l'entraînement hugging face
def trainer_compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average="macro"),
    }
