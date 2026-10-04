"""FastAPI bridge between assistant-ui's data stream and LangGraph."""

import asyncio
import logging
from collections.abc import AsyncIterator
from typing import Any

from assistant_stream import RunController, create_run
from assistant_stream.serialization import DataStreamResponse
from fastapi import FastAPI, HTTPException
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from app.config import Settings
from app.graph import create_graph
from app.models import create_chat_model

logger = logging.getLogger(__name__)


class ChatRequest(BaseModel):
    """The text conversation sent by assistant-ui's data-stream runtime."""

    model_config = ConfigDict(extra="ignore")

    messages: list[dict[str, Any]]


def _message_text(message: dict[str, Any]) -> str:
    """Read text from the assistant-ui / AI SDK generic message representation."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        if any(part.get("type") != "text" for part in content if isinstance(part, dict)):
            raise HTTPException(status_code=422, detail="Only text messages are supported.")
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and isinstance(part.get("text", ""), str)
        )
    raise HTTPException(status_code=422, detail="Only text messages are supported.")


def _messages_for_graph(messages: list[dict[str, Any]]) -> list[BaseMessage]:
    """Convert frontend messages to LangChain messages, dropping an empty run stub."""
    relevant = list(messages)
    if relevant and relevant[-1].get("role") == "assistant":
        if not _message_text(relevant[-1]).strip():
            relevant.pop()

    result: list[BaseMessage] = []
    for message in relevant:
        role = message.get("role")
        text = _message_text(message)
        if role == "user":
            result.append(HumanMessage(content=text))
        elif role == "assistant":
            result.append(AIMessage(content=text))
        else:
            raise HTTPException(
                status_code=422, detail="Only user and assistant messages are supported."
            )
    return result


def _text_from_chunk(chunk: AIMessageChunk) -> str:
    content = chunk.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            part["text"]
            for part in content
            if isinstance(part, dict)
            and part.get("type") == "text"
            and isinstance(part.get("text"), str)
        )
    return ""


async def _stream_graph(graph: Any, messages: list[BaseMessage]) -> AsyncIterator[str]:
    """Yield text deltas produced by LangGraph's messages stream mode."""
    async for chunk, _metadata in graph.astream({"messages": messages}, stream_mode="messages"):
        if isinstance(chunk, AIMessageChunk):
            text = _text_from_chunk(chunk)
            if text:
                yield text


def create_app(settings: Settings | None = None, model: BaseChatModel | None = None) -> FastAPI:
    """Create an app, allowing tests and other integrations to inject a chat model."""
    app_settings = settings or Settings()
    graph = create_graph(model) if model is not None else None

    app = FastAPI(title="Tutor API", version="0.1.0")
    app.state.settings = app_settings
    app.state.graph = graph

    @app.get("/api/health")
    async def health() -> dict[str, str | bool]:
        return {
            "status": "ok",
            "model_configured": app_settings.model_configured or model is not None,
        }

    @app.post("/api/chat")
    async def chat(request: ChatRequest) -> DataStreamResponse:
        selected_graph = app.state.graph
        if selected_graph is None:
            if not app_settings.model_configured:
                raise HTTPException(
                    status_code=503,
                    detail="Configure LLM_BASE_URL and LLM_MODEL in .env, then restart the app.",
                )
            selected_graph = create_graph(create_chat_model(app_settings))

        input_messages = _messages_for_graph(request.messages)
        if not input_messages:
            raise HTTPException(status_code=422, detail="Send a non-empty text message to start.")

        async def run(controller: RunController) -> None:
            try:
                async for text in _stream_graph(selected_graph, input_messages):
                    if controller.is_cancelled:
                        break
                    controller.append_text(text)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Chat model request failed")
                controller.add_error(
                    "The model request failed. Check LLM_BASE_URL, LLM_MODEL, "
                    "LLM_API_KEY, and the backend logs."
                )

        return DataStreamResponse(create_run(run))

    return app


app = create_app()
