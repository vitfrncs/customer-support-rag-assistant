"""Testes do pipeline de geração de respostas."""

from unittest.mock import MagicMock

import pytest

from customer_support_rag_assistant.pipeline import generate_support_draft
from customer_support_rag_assistant.rag.schemas import DraftResponse


def test_generate_support_draft(monkeypatch):
    retriever = MagicMock()
    retriever.search.return_value = [
        {
            "example_id": "ex_001",
            "instruction": "Where is my order?",
            "response": "I already checked your order.",
            "intent": "track_order",
            "category": "ORDER",
            "score": 0.91,
        },
        {
            "example_id": "ex_002",
            "instruction": "How can I track my package?",
            "response": "Your package has been shipped.",
            "intent": "track_order",
            "category": "ORDER",
            "score": 0.87,
        },
    ]

    structured_llm = MagicMock()
    structured_llm.invoke.return_value = DraftResponse(
        draft="Could you please provide your order reference?",
        information_needed=["Order reference"],
    )

    llm = MagicMock()
    llm.with_structured_output.return_value = structured_llm

    monkeypatch.setattr(
        "customer_support_rag_assistant.pipeline.create_llm",
        lambda: llm,
    )

    result = generate_support_draft(
        customer_message="My order hasn't arrived.",
        retriever=retriever,
    )

    assert result["selected_intent"] == "track_order"
    assert result["information_needed"] == ["Order reference"]
    assert len(result["retrieved_examples"]) == 2
    assert result["draft"] == "Could you please provide your order reference?"

    messages = structured_llm.invoke.call_args.args[0]

    assert "Where is my order?" in messages[1].content
    assert "I already checked your order." not in messages[1].content
    assert "Your package has been shipped." not in messages[1].content


def test_empty_customer_message():
    with pytest.raises(ValueError):
        generate_support_draft(
            customer_message="   ",
            retriever=MagicMock(),
        )


def test_invalid_top_k():
    with pytest.raises(ValueError):
        generate_support_draft(
            customer_message="Where is my order?",
            retriever=MagicMock(),
            top_k=0,
        )