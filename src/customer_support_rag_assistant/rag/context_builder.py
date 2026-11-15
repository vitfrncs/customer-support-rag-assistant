"""Prepara exemplos recuperados e orientações para a geração com RAG."""

import json
from collections import Counter
from pathlib import Path


def load_guidelines(path: str | Path) -> dict:
    """Carrega as orientações de atendimento do projeto."""
    with Path(path).open(encoding="utf-8") as file:
        return json.load(file)


def build_context(examples: list[dict], guidelines: dict) -> dict:
    """Organiza as solicitações semelhantes e seleciona orientações."""

    if not isinstance(examples, list):
        raise TypeError("Os exemplos devem formar uma lista.")

    if not isinstance(guidelines, dict):
        raise TypeError("As orientações devem formar um dicionário.")

    intents = [example["intent"] for example in examples]
    counts = Counter(intents)

    selected_intent = None

    if counts:
        intent, count = counts.most_common(1)[0]

        if count >= 2 and count > len(examples) / 2:
            if intent in guidelines["intents"]:
                selected_intent = intent

    rules = list(guidelines["general"])

    if selected_intent:
        rules.extend(guidelines["intents"][selected_intent])

    example_sections = [
        (
            f"[EXAMPLE {i}]\n"
            f"ID: {example['example_id']}\n"
            f"Intent: {example['intent']}\n"
            f"Customer request: {example['instruction']}"
        )
        for i, example in enumerate(examples, start=1)
    ]

    return {
        "examples": "\n\n".join(example_sections),
        "guidelines": "\n".join(f"- {rule}" for rule in rules),
        "selected_intent": selected_intent,
    }