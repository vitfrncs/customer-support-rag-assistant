"""Integra recuperação semântica e geração de respostas de atendimento."""

from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from customer_support_rag_assistant.rag.context_builder import (
    build_context,
    load_guidelines,
)
from customer_support_rag_assistant.rag.llm import create_llm
from customer_support_rag_assistant.rag.prompts import build_messages
from customer_support_rag_assistant.rag.schemas import DraftResponse


GUIDELINES_PATH = (
    Path(__file__).resolve().parents[2]
    / "data/knowledge/support_guidelines.json"
)


def generate_support_draft(customer_message: str,retriever,top_k: int = 3,) -> dict:
    """Gera um rascunho de atendimento a partir de exemplos recuperados."""

    if not isinstance(customer_message, str) or not customer_message.strip():
        raise ValueError("A mensagem do cliente não pode estar vazia.")

    if not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k deve ser um inteiro positivo.")

    examples = retriever.search(customer_message.strip(), top_k=top_k)
    guidelines = load_guidelines(GUIDELINES_PATH)

    context = build_context(examples, guidelines)
    messages = build_messages(customer_message, context)

    langchain_messages = [
        SystemMessage(content=messages[0]["content"]),
        HumanMessage(content=messages[1]["content"]),
    ]

    llm = create_llm().with_structured_output(
        DraftResponse,
        method="json_schema",
    )

    response = llm.invoke(langchain_messages)

    if not isinstance(response, DraftResponse):
        raise ValueError("O LLM retornou uma resposta inválida.")

    draft = response.draft.strip()

    if not draft:
        raise ValueError("O LLM retornou um rascunho vazio.")

    information_needed = list(dict.fromkeys(
        item.strip()
        for item in response.information_needed
        if item.strip()
    ))

    return {
        "draft": draft,
        "information_needed": information_needed,
        "selected_intent": context["selected_intent"],
        "retrieved_examples": [
            {
                "example_id": example["example_id"],
                "instruction": example["instruction"],
                "response": example["response"],
                "intent": example["intent"],
                "category": example["category"],
                "score": example["score"],
            }
            for example in examples
        ],
    }