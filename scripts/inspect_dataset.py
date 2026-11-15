"""Inspeção inicial do dataset."""

from datasets import load_dataset


DATASET_NAME = ("bitext/Bitext-customer-support-llm-chatbot-training-dataset")

RANDOM_STATE = 42

INTENTS_TO_INSPECT = [
    "track_order",
    "get_refund",
    "payment_issue",
    "contact_human_agent",
    "change_shipping_address",
]


def main() -> None:
    print("Carregando dataset...")

    dataset = load_dataset(
        DATASET_NAME,
        split="train",
    )

    df = dataset.to_pandas()

    print("\nINFORMAÇÕES GERAIS ")

    print(f"Registros: {len(df)}")
    print(f"Colunas: {list(df.columns)}")

    print("\nValores nulos:")
    print(df.isna().sum())

    print("\nSolicitações duplicadas:")
    print(df.duplicated(subset=["instruction"]).sum())

    # distribuição dos dados
    print("\nCATEGORIAS ")
    print(df["category"].value_counts().to_string())

    print("\nINTENÇÕES ")
    print(df["intent"].value_counts().to_string())

    # placeholders
    print("\nPLACEHOLDERS ")

    pattern = r"\{\{[^{}]+\}\}"

    instructions_with_placeholders = (
        df["instruction"]
        .fillna("")
        .str.contains(pattern, regex=True)
        .sum()
    )

    responses_with_placeholders = (
        df["response"]
        .fillna("")
        .str.contains(pattern, regex=True)
        .sum()
    )

    print(
        "Solicitações com placeholders:",
        instructions_with_placeholders,
    )

    print(
        "Respostas com placeholders:",
        responses_with_placeholders,
    )

    print("\nEXEMPLOS DE ATENDIMENTO ")

    for intent in INTENTS_TO_INSPECT:

        subset = df[df["intent"] == intent]

        if subset.empty:
            print(f"\nIntenção não encontrada: {intent}")
            continue

        examples = subset.sample(
            n=min(2, len(subset)),
            random_state=RANDOM_STATE,
        )

        print(f"\n--- INTENÇÃO: {intent} ---")

        for _, row in examples.iterrows():

            print("\nCliente:")
            print(row["instruction"])

            print("\nResposta:")
            print(str(row["response"])[:600])

            print("-" * 50)


if __name__ == "__main__":
    main()