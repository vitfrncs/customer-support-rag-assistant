"""Recupera exemplos de atendimento semanticamente semelhantes a uma consulta."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from customer_support_rag_assistant.retrieval.semantic import (
    compute_similarities,
    get_top_k_indices,
)


class SemanticRetriever:
    """Recupera exemplos de atendimento por similaridade semântica."""

    def __init__(self, dataset_path: str | Path, index_path: str | Path):
        self.examples = pd.read_csv(dataset_path)

        required_columns = {
            "example_id",
            "instruction",
            "response",
            "intent",
            "category",
        }

        if not required_columns.issubset(self.examples.columns):
            raise ValueError("O dataset não possui todas as colunas necessárias.")

        if self.examples.empty:
            raise ValueError("O dataset está vazio.")

        if self.examples["example_id"].isna().any():
            raise ValueError("Existem exemplos sem ID.")

        self.examples["example_id"] = self.examples["example_id"].astype(str)

        with np.load(index_path, allow_pickle=False) as index:
            self.example_embeddings = index["embeddings"]
            self.example_ids = index["example_ids"].astype(str)
            model_name = str(index["model_name"].item())

        if self.example_embeddings.ndim != 2:
            raise ValueError("O índice possui formato inválido.")

        if len(self.example_ids) != len(self.example_embeddings):
            raise ValueError("O número de embeddings e IDs não corresponde.")

        if len(set(self.example_ids)) != len(self.example_ids):
            raise ValueError("Existem IDs duplicados no índice.")

        if self.examples["example_id"].duplicated().any():
            raise ValueError("Existem IDs duplicados no dataset.")

        if set(self.example_ids) != set(self.examples["example_id"]):
            raise ValueError("O dataset e o índice possuem exemplos diferentes.")

        if not np.isfinite(self.example_embeddings).all():
            raise ValueError("O índice contém embeddings inválidos.")

        self.examples = self.examples.set_index("example_id", drop=False)

        self.model = SentenceTransformer(model_name, device="cpu")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A consulta deve ser um texto não vazio.")

        query_embedding = self.model.encode(
            f"query: {query.strip()}",
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return self.search_by_vector(query_embedding, top_k)

    def search_by_vector(
        self,
        query_embedding: np.ndarray,
        top_k: int = 3,
    ) -> list[dict]:

        query_embedding = np.asarray(query_embedding)

        if query_embedding.ndim != 1:
            raise ValueError("O vetor de consulta deve ter uma dimensão.")

        if len(query_embedding) != self.example_embeddings.shape[1]:
            raise ValueError("Dimensão incompatível com o índice.")

        if (
            not np.issubdtype(query_embedding.dtype, np.number)
            or np.issubdtype(query_embedding.dtype, np.complexfloating)
        ):
            raise ValueError("O vetor deve conter valores numéricos reais.")

        if not np.isfinite(query_embedding).all():
            raise ValueError("O vetor contém valores inválidos.")

        if not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k deve ser um inteiro positivo.")

        norm = np.linalg.norm(query_embedding.astype(np.float64))

        if norm == 0:
            raise ValueError("O vetor não pode ser nulo.")

        query_embedding = query_embedding / norm

        scores = compute_similarities(
            query_embedding,
            self.example_embeddings,
        )

        top_indices = get_top_k_indices(
            scores,
            min(top_k, len(self.example_ids)),
        )

        results = []

        for index in top_indices:
            example_id = self.example_ids[index]
            example = self.examples.loc[example_id]

            results.append({
                "example_id": str(example_id),
                "instruction": example["instruction"],
                "response": example["response"],
                "intent": example["intent"],
                "category": example["category"],
                "score": float(scores[index]),
            })

        return results