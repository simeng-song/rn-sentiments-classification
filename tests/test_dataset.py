import pandas as pd

from src.amazon_sentiment.dataset import split_dataset


def test_split_dataset_creates_expected_files(tmp_path):
    labels = ["negative"] * 10 + ["positive"] * 10 + ["neutre"] * 10
    df = pd.DataFrame(
        {
            "rating": [1] * 10 + [5] * 10 + [3] * 10,
            "clean_text": [f"text {i}" for i in range(30)],
            "label": labels,
        }
    )
    input_csv = tmp_path / "dataset.csv"
    output_dir = tmp_path / "splits"
    df.to_csv(input_csv, index=False)

    paths = split_dataset(input_csv, output_dir, "demo", random_state=42)

    assert paths["train"].exists()
    assert paths["dev"].exists()
    assert paths["test"].exists()
