"""Avalia a recuperação semântica utilizando as intenções do Bitext."""

from pathlib import Path

import pandas as pd

from customer_support_rag_assistant.retrieval.retriever import SemanticRetriever


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data/processed"
RESULTS_DIR = ROOT_DIR / "results"

TOP_K = 3


def main():
    index = pd.read_csv(DATA_DIR / "index_examples.csv")
    queries = pd.read_csv(DATA_DIR / "eval_queries.csv")

    retriever = SemanticRetriever(
        dataset_path=DATA_DIR / "index_examples.csv",
        index_path=DATA_DIR / "semantic_index_e5.npz",
    )

    relevant_counts = index["intent"].value_counts()
    results = []

    for query in queries.itertuples(index=False):
        retrieved = retriever.search(query.instruction, top_k=TOP_K)

        matches = [
            example["intent"] == query.intent
            for example in retrieved
        ]

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
    metrics = evaluation.select_dtypes(include="number").mean()

    print("\nRETRIEVAL EVALUATION ")
    print(f"Queries: {len(evaluation)}")

    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
        

    print("\nRESULTS BY INTENT ")
    by_intent = evaluation.groupby("intent").mean(numeric_only=True)
    print(by_intent.round(3).to_string())

    RESULTS_DIR.mkdir(exist_ok=True)
    evaluation.to_csv(RESULTS_DIR / "retrieval_e5_queries.csv", index=False)
    by_intent.to_csv(RESULTS_DIR / "retrieval_e5_by_intent.csv")


if __name__ == "__main__":
    main()