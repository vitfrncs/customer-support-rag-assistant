"""Define o formato das respostas estruturadas geradas pelo LLM."""

from pydantic import BaseModel, Field


class DraftResponse(BaseModel):
    """Representa uma sugestão de atendimento para revisão humana."""

    draft: str = Field(
        min_length=1,
        description="Suggested response to the customer.",
    )

    information_needed: list[str] = Field(
        default_factory=list,
        description="Information that the support agent needs to verify.",
    )