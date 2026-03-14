"""Résumer et visualiser les résultats"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .evaluation import metrics_summary_frame


# regroupe plusieurs fichiers de métriques dans un seul tableau
def build_summary(metrics_files, output_csv=None):
    summary = metrics_summary_frame(metrics_files)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_path, index=False)
    return summary


# créer des graphiques de comparaison pour l'accuracy et la macro-f1
def plot_summary(summary, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_files = []

    # on garde une couleur par famille de modèle pour que la lecture soit plus simple
    colors = ["#1f77b4" if model == "logreg" else "#ff7f0e" for model in summary["model_type"]]

    for metric_name, filename, ylabel in [
        ("accuracy", "accuracy_comparison.png", "Exactitude"),
        ("f1_macro", "f1_macro_comparison.png", "F1 macro"),
    ]:
        plt.figure(figsize=(10, 5))
        plt.bar(summary["run_name"], summary[metric_name], color=colors)
        plt.xticks(rotation=25, ha="right")
        plt.ylabel(ylabel)
        plt.title(f"Comparaison de {ylabel} sur le jeu de test")
        plt.tight_layout()
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()
        generated_files.append(output_path)

    return generated_files
