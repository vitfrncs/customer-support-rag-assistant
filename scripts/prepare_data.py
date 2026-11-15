"""Prepara e divide o dataset Bitext para recuperação e avaliação."""

from pathlib import Path

import pandas as pd
from datasets import load_dataset


DATASET_NAME = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"

INDEX_PER_INTENT = 25
EVAL_PER_INTENT = 5
SEED = 42

COLUMNS = ["example_id", "instruction", "response", "intent", "category"]


def prepare_data() -> None:
    df = load_dataset(DATASET_NAME, split="train").to_pandas()

    required = COLUMNS[1:]
    df = df.dropna(subset=required).copy()

    for column in required:
        df[column] = df[column].str.strip()

    df = df[df[required].ne("").all(axis=1)].copy()
    df["example_id"] = [f"ex_{i:05d}" for i in df.index]

    # Remove requests identicos antes de dividir para evitar data leakage
    df["normalized_instruction"] = (
        df["instruction"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )

    df = df.drop_duplicates(subset="normalized_instruction")

    index_samples = []
    eval_samples = []

    for intent, group in df.groupby("intent"):
        sample_size = INDEX_PER_INTENT + EVAL_PER_INTENT

        if len(group) < sample_size:
            raise ValueError(f"Not enough examples for intent: {intent}")

        sample = group.sample(n=sample_size, random_state=SEED)

        index_samples.append(sample.iloc[:INDEX_PER_INTENT])
        eval_samples.append(sample.iloc[INDEX_PER_INTENT:])

    index_df = pd.concat(index_samples, ignore_index=True)
    eval_df = pd.concat(eval_samples, ignore_index=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    index_df[COLUMNS].to_csv(OUTPUT_DIR / "index_examples.csv", index=False)
    eval_df[COLUMNS].to_csv(OUTPUT_DIR / "eval_queries.csv", index=False)

    print(f"Index: {len(index_df)} examples")
    print(f"Evaluation: {len(eval_df)} examples")


if __name__ == "__main__":
    prepare_data()