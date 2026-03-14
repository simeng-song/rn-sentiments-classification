"""interface en ligne de commande du projet"""

import argparse
import json
from pathlib import Path

from .dataset import extract_fields_from_jsonl, split_dataset
from .evaluation import load_metrics
from .labeling import label_jsonl
from .models import train_bert_model, train_logreg_model
from .predict import predict_csv, predict_texts
from .preprocessing import clean_jsonl, sample_jsonl
from .visualize import build_summary, plot_summary


# ajoute les arguments communs aux commandes d'entraînement
def _add_common_split_arguments(parser):
    parser.add_argument("--train", required=True, help="Chemin vers le fichier CSV d'entraînement.")
    parser.add_argument("--dev", required=True, help="Chemin vers le fichier CSV de validation.")
    parser.add_argument("--test", required=True, help="Chemin vers le fichier CSV de test.")
    parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Garde la classe neutre au lieu de faire seulement du binaire.",
    )


# construit le parseur principal de la cli
def build_parser():
    parser = argparse.ArgumentParser(description="CLI pour l'analyse de sentiments sur des avis Amazon")
    subparsers = parser.add_subparsers(dest="command", required=True)

    preprocess_parser = subparsers.add_parser(
        "preprocess",
        help="Prépare les données : échantillon, nettoyage, labels, CSV et split.",
    )
    preprocess_parser.add_argument("--input-jsonl", required=True, help="Fichier JSONL brut de départ.")
    preprocess_parser.add_argument("--sample-output", help="Fichier JSONL de sortie si on fait un échantillon.")
    preprocess_parser.add_argument("--sample-size", type=int, default=0, help="Nombre d'avis à garder.")
    preprocess_parser.add_argument("--clean-output", required=True, help="Fichier JSONL nettoyé.")
    preprocess_parser.add_argument("--labeled-output", required=True, help="Fichier JSONL avec labels.")
    preprocess_parser.add_argument("--csv-output", required=True, help="Fichier CSV final.")
    preprocess_parser.add_argument("--split-output-dir", required=True, help="Dossier de sortie pour train/dev/test.")
    preprocess_parser.add_argument("--prefix", required=True, help="Préfixe des noms de fichiers.")
    preprocess_parser.add_argument("--seed", type=int, default=42, help="Graine aléatoire.")
    preprocess_parser.add_argument(
        "--include-neutral",
        action="store_true",
        help="Garde la classe neutre pour les expériences à trois classes.",
    )

    logreg_parser = subparsers.add_parser("train-logreg", help="Entraîne la baseline en régression logistique.")
    _add_common_split_arguments(logreg_parser)
    logreg_parser.add_argument("--output-dir", default="outputs/models", help="Dossier où sauver le modèle.")
    logreg_parser.add_argument("--metrics-dir", default="outputs/metrics", help="Dossier où sauver les métriques.")

    bert_parser = subparsers.add_parser("train-bert", help="Entraîne DistilBERT sur les données.")
    _add_common_split_arguments(bert_parser)
    bert_parser.add_argument("--output-dir", default="outputs/models", help="Dossier où sauver le modèle.")
    bert_parser.add_argument("--metrics-dir", default="outputs/metrics", help="Dossier où sauver les métriques.")
    bert_parser.add_argument("--model-name", default="distilbert-base-uncased", help="Nom du modèle Hugging Face.")
    bert_parser.add_argument("--epochs", type=int, default=2, help="Nombre d'époques.")
    bert_parser.add_argument("--max-steps", type=int, default=800, help="Nombre maximum de steps.")

    evaluate_parser = subparsers.add_parser("evaluate", help="Affiche un fichier JSON de métriques.")
    evaluate_parser.add_argument("--metrics-file", required=True, help="Fichier JSON à afficher.")

    visualize_parser = subparsers.add_parser("visualize", help="Crée un tableau récapitulatif et des graphiques.")
    visualize_parser.add_argument("--metrics-files", nargs="+", required=True, help="Liste des fichiers JSON.")
    visualize_parser.add_argument("--summary-csv", default="outputs/metrics/summary.csv", help="Chemin du CSV récapitulatif.")
    visualize_parser.add_argument("--output-dir", default="outputs/plots", help="Dossier des graphiques.")

    predict_parser = subparsers.add_parser("predict", help="Prédit le sentiment d'un texte ou d'un CSV.")
    predict_parser.add_argument("--model-type", choices=["logreg", "bert"], required=True, help="Famille de modèle.")
    predict_parser.add_argument("--model-path", required=True, help="Chemin vers le modèle sauvegardé.")
    predict_group = predict_parser.add_mutually_exclusive_group(required=True)
    predict_group.add_argument("--text", help="Phrase unique à classer.")
    predict_group.add_argument("--input-csv", help="Fichier CSV qui contient une colonne de texte.")
    predict_parser.add_argument("--text-column", default="text", help="Nom de la colonne texte dans le CSV.")
    predict_parser.add_argument(
        "--output-csv",
        default="outputs/predictions/predictions.csv",
        help="Chemin du CSV de sortie pour les prédictions.",
    )

    return parser


# lance la bonne commande selon les arguments donnés
def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "preprocess":
        input_jsonl = Path(args.input_jsonl)
        cleaned_input = input_jsonl
        # si on demande un échantillon, on passe d'abord par cette étape
        if args.sample_size:
            if not args.sample_output:
                parser.error("--sample-output is required when --sample-size is used.")
            sample_jsonl(input_jsonl, args.sample_output, args.sample_size, seed=args.seed)
            cleaned_input = Path(args.sample_output)

        clean_jsonl(cleaned_input, args.clean_output)
        label_jsonl(args.clean_output, args.labeled_output, include_neutral=args.include_neutral)
        extract_fields_from_jsonl(args.labeled_output, args.csv_output)
        split_paths = split_dataset(args.csv_output, args.split_output_dir, args.prefix, random_state=args.seed)
        print(f"Prétraitement terminé. Fichiers créés : {split_paths}")
        return

    if args.command == "train-logreg":
        # ici on entraîne la baseline puis on écrit aussi les métriques dans outputs/metrics
        model_metrics = train_logreg_model(
            args.train,
            args.dev,
            args.test,
            args.output_dir,
            include_neutral=args.include_neutral,
        )
        metrics_path = Path(args.metrics_dir) / f"{model_metrics['run_name']}.json"
        Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(model_metrics["test"], ensure_ascii=False, indent=2))
        return

    if args.command == "train-bert":
        # même logique ici, mais avec distilbert
        model_metrics = train_bert_model(
            args.train,
            args.dev,
            args.test,
            args.output_dir,
            include_neutral=args.include_neutral,
            model_name=args.model_name,
            num_train_epochs=args.epochs,
            max_steps=args.max_steps,
        )
        metrics_path = Path(args.metrics_dir) / f"{model_metrics['run_name']}.json"
        Path(args.metrics_dir).mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(json.dumps(model_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(model_metrics["test"], ensure_ascii=False, indent=2))
        return

    if args.command == "evaluate":
        # cette commande sert juste à relire un fichier de métriques déjà produit
        metrics = load_metrics(args.metrics_file)
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
        return

    if args.command == "visualize":
        # on regroupe les métriques puis on génère les graphiques
        summary = build_summary(args.metrics_files, args.summary_csv)
        generated = plot_summary(summary, args.output_dir)
        print(f"Résumé écrit dans {args.summary_csv}")
        print(f"Graphiques générés : {[str(path) for path in generated]}")
        return

    if args.command == "predict":
        if args.text:
            predictions = predict_texts(args.model_type, args.model_path, [args.text])
            print(predictions[0])
            return
        # sinon on lit un csv entier et on ajoute une colonne de prédiction
        saved_path = predict_csv(
            args.model_type,
            args.model_path,
            args.input_csv,
            args.output_csv,
            text_column=args.text_column,
        )
        print(f"Prédictions écrites dans {saved_path}")