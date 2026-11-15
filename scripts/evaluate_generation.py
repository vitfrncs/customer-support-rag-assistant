"""Executa casos de avaliação e salva os rascunhos gerados pelo RAG."""

import json
from pathlib import Path
import os
import pandas as pd

from customer_support_rag_assistant.pipeline import generate_support_draft
from customer_support_rag_assistant.retrieval.retriever import SemanticRetriever


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data/processed"
CASES_PATH = ROOT_DIR / "data/evaluation/generation_cases.csv"
OUTPUT_PATH = ROOT_DIR / "results/generation_evaluation.csv"


def main():
    if OUTPUT_PATH.exists():
        raise FileExistsError(
            f"{OUTPUT_PATH} já existe. Renomeie o arquivo antes de executar novamente."
        )

    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")):
        raise EnvironmentError("Gemini API key is not configured.")

    cases = pd.read_csv(CASES_PATH)
    retriever = SemanticRetriever(
        dataset_path=DATA_DIR / "index_examples.csv",
        index_path=DATA_DIR / "semantic_index_e5.npz",
    )

    results = []
    OUTPUT_PATH.parent.mkdir(exist_ok=True)

    for case in cases.itertuples(index=False):
        print(f"Evaluating case {case.case_id}")

        row = {
            "case_id": case.case_id,
            "customer_message": case.customer_message,
            "expected_language": case.expected_language,
            "focus": getattr(case, "focus", ""),
            "selected_intent": "",
            "retrieved_intents": "",
            "draft": "",
            "information_needed": "",
            "language_ok": "",
            "helpful": "",
            "guidelines_ok": "",
            "no_unverified_actions": "",
            "information_needed_ok": "",
            "notes": "",
            "error": "",
        }

        try:
            response = generate_support_draft(
                customer_message=case.customer_message,
                retriever=retriever,
            )

            row["selected_intent"] = response["selected_intent"]
            row["retrieved_intents"] = json.dumps(
                [example["intent"] for example in response["retrieved_examples"]]
            )
            row["draft"] = response["draft"]
            row["information_needed"] = json.dumps(
                response["information_needed"],
                ensure_ascii=False,
            )

        except Exception as error:
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"Error: {error}")

        results.append(row)
        pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)

    print(f"\nResults saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()