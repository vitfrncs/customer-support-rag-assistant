"""Interface Streamlit do assistente de atendimento."""

import logging
from pathlib import Path

import streamlit as st

from customer_support_rag_assistant.pipeline import generate_support_draft
from customer_support_rag_assistant.retrieval.retriever import SemanticRetriever


DATA_DIR = Path(__file__).resolve().parent / "data/processed"


@st.cache_resource
def load_retriever():
    return SemanticRetriever(
        dataset_path=DATA_DIR / "index_examples.csv",
        index_path=DATA_DIR / "semantic_index_e5.npz",
    )


def main():
    st.set_page_config(
        page_title="Customer Support RAG Assistant",
        page_icon="💬",
    )

    st.title("Customer Support RAG Assistant")
    st.caption("Synthetic data · Human review required")

    with st.form("support_form"):
        message = st.text_area(
            "Customer message",
            placeholder="My order hasn't arrived yet. Can you help me?",
            height=120,
        )
        submitted = st.form_submit_button("Generate response")

    if submitted:
        st.session_state.pop("result", None)

        if not message.strip():
            st.warning("Please enter a customer message.")
        else:
            try:
                with st.spinner("Generating response..."):
                    result = generate_support_draft(message, load_retriever())
                    st.session_state["result"] = result
                    st.session_state["draft"] = result["draft"]

            except Exception:
                logging.exception("Response generation failed")
                st.error("Could not generate a response. Check the terminal.")

    result = st.session_state.get("result")

    if result is None:
        return

    st.subheader("Suggested response")
    st.text_area("Draft", key="draft", height=150)
    st.caption("Review and edit before sending.")

    if result["information_needed"]:
        st.subheader("Information needed")

        for item in result["information_needed"]:
            st.write(f"- {item}")

    with st.expander("Similar cases"):
        st.caption(
            "Synthetic examples for reference only. "
            "Responses may contain unverified claims."
        )

        for example in result["retrieved_examples"]:
            st.markdown("**Customer:**")
            st.write(example["instruction"])

            st.markdown("**Response:**")
            st.write(example["response"])

            st.caption(
                f"{example['intent']} · "
                f"Similarity: {example['score']:.3f}"
            )
            st.divider()


if __name__ == "__main__":
    main()