"""Prétraitement pour les avis Amazon. (échantillonnage aléatoire, nettoyage et normalisation)"""

import json
import random
import re
import string
from pathlib import Path

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


def ensure_nltk_resources():
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("corpora/stopwords", "stopwords"),
    ]
    for resource_path, resource_name in resources:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(resource_name, quiet=True)
            except Exception:
                pass


def sample_jsonl(input_path, output_path, sample_size, seed=42):
    """Prend un échantillon aléatoire dans un fichier JSONL."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    # Lecture de toutes les lignes du fichier
    with input_path.open("r", encoding="utf-8") as infile:
        lines = infile.readlines()
	# Initialisation du générateur aléatoire
    random.seed(seed)
    sample = random.sample(lines, min(sample_size, len(lines)))
    # Ecriture du nouvel échantillon
    with output_path.open("w", encoding="utf-8") as outfile:
        outfile.writelines(sample)
    return len(sample)


def clean_text(text):
    """Nettoie et normalise un texte pour l'entraînement."""
    ensure_nltk_resources()
    text = "" if text is None else str(text)
    # Conversion en minuscules
    lowered = text.lower()
    # Remplacement de la ponctuation par des espaces
    normalized = re.sub(r"[^\w\s]", " ", lowered)
    
    try:
        stop_words = set(stopwords.words("english"))
    except LookupError:
        stop_words = {"a", "an", "and", "the", "is", "it", "of", "to", "was", "very"}
    punct = set(string.punctuation)
    
    try:
    	# Tokenisation du texte
        tokens = word_tokenize(normalized)
    except LookupError:
        # En dernier recours, on coupe simplement sur les espaces.
        tokens = normalized.split()
    
    # Suppression des stopwords et de la ponctuation
    filtered = [
    	token for token in tokens 
    	if token not in stop_words and token not in punct
    ]
    return " ".join(filtered)


def clean_jsonl(input_path, output_path, text_field="text"):
    input_path = Path(input_path)
    output_path = Path(output_path)
    count = 0
	
	# Lecture du fichier source et écriture dans un nouveau fichier
    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
        	# Conversion de la ligne JSON en dictionnaire Python
            record = json.loads(line)
            record["clean_text"] = clean_text(record.get(text_field, ""))
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1

    return count
