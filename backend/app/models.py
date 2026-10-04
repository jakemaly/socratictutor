"""Model adapter construction kept separate from the LangGraph workflow."""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI

from app.config import Settings


def create_chat_model(settings: Settings) -> BaseChatModel:
    """Build a standard Chat Completions client for an OpenAI-compatible URL."""
    if not settings.model_configured:
        raise ValueError("Set LLM_BASE_URL and LLM_MODEL before starting a chat.")

    api_key = settings.llm_api_key.get_secret_value().strip() if settings.llm_api_key else ""
    return ChatOpenAI(
        model=settings.llm_model,
        base_url=settings.llm_base_url,
        api_key=api_key or "not-required",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
    )
