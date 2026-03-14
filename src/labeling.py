"""Créer les labels de sentiment."""

import json
from pathlib import Path
from typing import Iterable


def label_from_rating(rating, include_neutral=True):
    """Transforme une note Amazon en label de sentiment."""
    if rating in {4, 4.0, 5, 5.0}:
        return "positive"
    if rating in {1, 1.0, 2, 2.0}:
        return "negative"
    if rating in {3, 3.0}:
        return "neutre" if include_neutral else None
    return None


def iter_labeled_records(
    records: Iterable[dict], include_neutral=True
):
    """Parcourt des enregistrements et ajoute un label quand c'est possible."""
    for record in records:
        labeled = dict(record)
        labeled["label"] = label_from_rating(record.get("rating"), include_neutral)
        if labeled["label"] is not None:
            yield labeled


def label_jsonl(input_path, output_path, include_neutral=True):
    """Lit un JSONL, ajoute les labels, puis écrit le résultat."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    kept = 0

    with input_path.open("r", encoding="utf-8") as infile, output_path.open(
        "w", encoding="utf-8"
    ) as outfile:
        for line in infile:
            record = json.loads(line)
            label = label_from_rating(record.get("rating"), include_neutral)
            # Si la note ne correspond à aucun cas, on saute la ligne.
            if label is None:
                continue
            record["label"] = label
            outfile.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1

    return kept
