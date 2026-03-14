"""Entraîner et recharger les modèles du projet."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .evaluation import compute_metrics_from_predictions, prediction_frame, save_metrics, trainer_compute_metrics

# charge un csv et retire la classe neutre (si besoin
def load_split_csv(file_path, include_neutral=True):
    df = pd.read_csv(file_path)
    if not include_neutral:
        df = df[df["label"] != "neutre"]
    texts = df["clean_text"].fillna("").astype(str).tolist()
    labels = df["label"].tolist()
    return texts, labels

# récupère un nom de dataset
def infer_dataset_name(train_file):
    train_path = str(train_file)
    name = Path(train_path).stem.replace("_train", "")
    return name

# renommer (2 va etre un peu ambiguite
def format_dataset_slug(train_file):
    dataset_name = infer_dataset_name(train_file)
    lowered = dataset_name.lower()
    lowered = lowered.replace("books", "books").replace("kindle", "kindle")
    lowered = lowered.replace("_2", "_2classes").replace("_3", "_3classes")
    return lowered


###---------------------TF-IDF + régression logistique-------------------------###
def train_logreg_model(
    train_file,
    dev_file,
    test_file,
    output_dir,
    include_neutral=True,
    max_features=5000,
    max_iter=1000,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # charge les trois splits
    x_train, y_train = load_split_csv(train_file, include_neutral)
    x_dev, y_dev = load_split_csv(dev_file, include_neutral)
    x_test, y_test = load_split_csv(test_file, include_neutral)

    # encode les labels en nombres
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(y_train)
    y_dev_encoded = label_encoder.transform(y_dev)
    y_test_encoded = label_encoder.transform(y_test)

    # reste sur une vectorisation simple
    vectorizer = TfidfVectorizer(max_features=max_features)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_dev_tfidf = vectorizer.transform(x_dev)
    x_test_tfidf = vectorizer.transform(x_test)

    model = LogisticRegression(max_iter=max_iter)
    model.fit(x_train_tfidf, y_train_encoded)

    dev_pred = model.predict(x_dev_tfidf)
    test_pred = model.predict(x_test_tfidf)
    label_names = list(label_encoder.classes_)

    dataset_name = infer_dataset_name(train_file)
    dataset_slug = format_dataset_slug(train_file)
    run_name = f"logreg_{dataset_slug}"
    metrics = {
        "run_name": run_name,
        "model_type": "logreg",
        "dataset_name": dataset_name,
        "include_neutral": include_neutral,
        "labels": label_names,
        "dev": compute_metrics_from_predictions(y_dev_encoded, dev_pred, label_names),
        "test": compute_metrics_from_predictions(y_test_encoded, test_pred, label_names),
    }
    save_metrics(metrics, output_dir / f"{run_name}.json")

    # on garde aussi le vectorizer et l'encodeur, sinon le modèle seul ne suffit pas
    joblib.dump(
        {"model": model, "vectorizer": vectorizer, "label_encoder": label_encoder},
        output_dir / f"{run_name}.joblib",
    )

    predictions = prediction_frame(
        x_test,
        [label_names[index] for index in y_test_encoded],
        [label_names[index] for index in test_pred],
    )
    predictions.to_csv(output_dir / f"{run_name}_test_predictions.csv", index=False)
    return metrics


# Transforme un split CSV en Dataset compatible avec Hugging Face.
def _build_hf_dataset(
    file_path, label_encoder, include_neutral
):
    df = pd.read_csv(file_path)
    if not include_neutral:
        df = df[df["label"] != "neutre"].copy()
    df["clean_text"] = df["clean_text"].fillna("").astype(str)

    # créer l'encodeur sur le train, puis le réutiliser sur dev et test.
    if label_encoder is None:
        label_encoder = LabelEncoder()
        df["labels"] = label_encoder.fit_transform(df["label"])
    else:
        df["labels"] = label_encoder.transform(df["label"])

    dataset = Dataset.from_pandas(df[["clean_text", "labels"]], preserve_index=False)
    return dataset, df, label_encoder



###---------------------DistilBERT-------------------------###
def train_bert_model(
    train_file,
    dev_file,
    test_file,
    output_dir,
    include_neutral=True,
    model_name="distilbert-base-uncased",
    num_train_epochs=2,
    max_steps=800,
    learning_rate=2e-5,
    per_device_train_batch_size=12,
    per_device_eval_batch_size=12,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # prépare les trois splits
    train_dataset, train_df, label_encoder = _build_hf_dataset(train_file, None, include_neutral)
    dev_dataset, dev_df, label_encoder = _build_hf_dataset(dev_file, label_encoder, include_neutral)
    test_dataset, test_df, label_encoder = _build_hf_dataset(test_file, label_encoder, include_neutral)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(batch["clean_text"], truncation=True, padding="max_length", max_length=128)

    # tokenise tout avant l'entraînement
    train_dataset = train_dataset.map(tokenize, batched=True)
    dev_dataset = dev_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    dev_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(label_encoder.classes_)
    )

    artifacts_dir = output_dir / "artifacts"
    # on garde une écriture compatible avec plusieurs versions de transformers, parce que l'environnement peut changer selon la machine.
    training_args_kwargs = {
        "output_dir": str(artifacts_dir),
        "logging_strategy": "epoch",
        "save_strategy": "epoch",
        "save_total_limit": 1,
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1_macro",
        "num_train_epochs": num_train_epochs,
        "max_steps": max_steps,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "learning_rate": learning_rate,
        "weight_decay": 0.01,
        "report_to": "none",
        "logging_dir": str(artifacts_dir / "logs"),
    }
    try:
        training_args = TrainingArguments(
            eval_strategy="epoch",
            **training_args_kwargs,
        )
    except TypeError:
        training_args = TrainingArguments(
            evaluation_strategy="epoch",
            **training_args_kwargs,
        )

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": dev_dataset,
        "compute_metrics": trainer_compute_metrics,
    }
    # pareil ici : certaines versions attendent `processing_class`,
    # d'autres `tokenizer`.
    try:
        trainer = Trainer(
            processing_class=tokenizer,
            **trainer_kwargs,
        )
    except TypeError:
        trainer = Trainer(
            tokenizer=tokenizer,
            **trainer_kwargs,
        )
    trainer.train()

    # évalue ensuite sur dev et test pour récupérer des métriques claires.
    dev_predictions = trainer.predict(dev_dataset)
    test_predictions = trainer.predict(test_dataset)
    dev_pred = np.argmax(dev_predictions.predictions, axis=1)
    test_pred = np.argmax(test_predictions.predictions, axis=1)
    label_names = list(label_encoder.classes_)

    dataset_name = infer_dataset_name(train_file)
    dataset_slug = format_dataset_slug(train_file)
    run_name = f"distilbert_{dataset_slug}"
    metrics = {
        "run_name": run_name,
        "model_type": "bert",
        "dataset_name": dataset_name,
        "include_neutral": include_neutral,
        "transformer_model": model_name,
        "labels": label_names,
        "dev": compute_metrics_from_predictions(dev_df["labels"], dev_pred, label_names),
        "test": compute_metrics_from_predictions(test_df["labels"], test_pred, label_names),
    }
    save_metrics(metrics, output_dir / f"{run_name}.json")

    # sauvegarde aussi les prédictions du test pour pouvoir les relire plus tard.
    predictions = prediction_frame(
        test_df["clean_text"],
        [label_names[index] for index in test_df["labels"]],
        [label_names[index] for index in test_pred],
    )
    predictions.to_csv(output_dir / f"{run_name}_test_predictions.csv", index=False)

    export_dir = output_dir / run_name
    trainer.save_model(str(export_dir))
    tokenizer.save_pretrained(str(export_dir))
    with (export_dir / "label_mapping.json").open("w", encoding="utf-8") as outfile:
        json.dump({label: int(index) for index, label in enumerate(label_names)}, outfile, indent=2)

    return metrics


# Recharge tout ce qu'il faut pour utiliser la baseline
def load_logreg_artifact(artifact_path):
    return joblib.load(artifact_path)

# Recharge un modèle DistilBERT déjà sauvegardé.
def load_bert_artifact(model_dir):
    model_dir = Path(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    mapping_path = model_dir / "label_mapping.json"
    with mapping_path.open("r", encoding="utf-8") as infile:
        label_mapping = json.load(infile)
    inverse_mapping = {int(index): label for label, index in label_mapping.items()}
    return model, tokenizer, inverse_mapping
