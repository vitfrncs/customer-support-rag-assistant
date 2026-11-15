"""Avalia a recuperação lexical com BM25 e compara seus resultados com o E5."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data/processed"
RESULTS_DIR = ROOT_DIR / "results"

TOP_K = 3
METRICS = ["precision_at_3", "recall_at_3", "hit_at_3", "mrr_at_3"]


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def main():
    examples = pd.read_csv(DATA_DIR / "index_examples.csv")
    queries = pd.read_csv(DATA_DIR / "eval_queries.csv")

    corpus = [tokenize(text) for text in examples["instruction"]]
    bm25 = BM25Okapi(corpus)

    relevant_counts = examples["intent"].value_counts()
    results = []

    for query in queries.itertuples(index=False):
        scores = bm25.get_scores(tokenize(query.instruction))
        indices = np.argsort(-scores, kind="stable")[:TOP_K]

        retrieved = examples.iloc[indices]
        matches = (retrieved["intent"] == query.intent).tolist()

        first_relevant = next(
            (rank for rank, match in enumerate(matches, start=1) if match),
            None,
        )

        results.append({
            "example_id": query.example_id,
            "intent": query.intent,
            "precision_at_3": sum(matches) / TOP_K,
            "recall_at_3": sum(matches) / relevant_counts[query.intent],
            "hit_at_3": int(any(matches)),
            "mrr_at_3": 1 / first_relevant if first_relevant else 0,
        })

    evaluation = pd.DataFrame(results)
    by_intent = evaluation.groupby("intent")[METRICS].mean()

    RESULTS_DIR.mkdir(exist_ok=True)

    evaluation.to_csv(
        RESULTS_DIR / "retrieval_bm25_queries.csv",
        index=False,
    )

    by_intent.to_csv(
        RESULTS_DIR / "retrieval_bm25_by_intent.csv",
    )

    e5_path = RESULTS_DIR / "retrieval_e5_queries.csv"

    if not e5_path.exists():
        raise FileNotFoundError(
            "Execute a avaliação do E5 antes de gerar a comparação."
        )

    e5 = pd.read_csv(e5_path)

    if set(evaluation["example_id"]) != set(e5["example_id"]):
        raise ValueError("BM25 e E5 foram avaliados em consultas diferentes.")

    comparison = pd.DataFrame({
        "BM25": evaluation[METRICS].mean(),
        "E5": e5[METRICS].mean(),
    }).T

    comparison.index.name = "retriever"

    comparison.to_csv(RESULTS_DIR / "retrieval_comparison.csv")

    print("\nRETRIEVAL COMPARISON ")
    print(comparison.round(4).to_string())

    print("\nBM25 RESULTS BY INTENT ")
    print(by_intent.round(3).to_string())


if __name__ == "__main__":
    main()