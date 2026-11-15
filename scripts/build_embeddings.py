"""Gera e armazena embeddings E5 das solicitações de atendimento."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


ROOT_DIR = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT_DIR / "data/processed/index_examples.csv"
INDEX_PATH = ROOT_DIR / "data/processed/semantic_index_e5.npz"

MODEL_NAME = "intfloat/multilingual-e5-small"


def main():
    examples = pd.read_csv(DATASET_PATH)

    if examples.empty:
        raise ValueError("O dataset está vazio.")

    if examples["example_id"].isna().any():
        raise ValueError("Existem exemplos sem ID.")

    if examples["example_id"].duplicated().any():
        raise ValueError("Existem IDs duplicados.")

    if examples["instruction"].isna().any():
        raise ValueError("Existem solicitações sem texto.")

    if examples["instruction"].str.strip().eq("").any():
        raise ValueError("Existem solicitações vazias.")

    texts = [
        f"passage: {instruction}"
        for instruction in examples["instruction"]
    ]

    model = SentenceTransformer(MODEL_NAME, device="cpu")

    embeddings = model.encode(
        texts,
        batch_size=8,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    example_ids = np.asarray(
        examples["example_id"].astype(str).tolist(),
        dtype=str,
    )

    np.savez_compressed(
        INDEX_PATH,
        embeddings=embeddings,
        example_ids=example_ids,
        model_name=np.array(MODEL_NAME),
    )

    print("Índice semântico criado com sucesso!!")
    print(f"Total de exemplos: {len(example_ids)}")
    print(f"Formato dos embeddings: {embeddings.shape}")
    print(f"Índice salvo em: {INDEX_PATH}")


if __name__ == "__main__":
    main()