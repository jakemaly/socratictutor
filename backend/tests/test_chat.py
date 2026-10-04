"""End-to-end checks for the FastAPI, LangGraph, and assistant-ui stream bridge."""

import asyncio
import json

import httpx
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from app.config import Settings
from app.graph import SYSTEM_PROMPT, create_graph
from app.main import ChatRequest, _messages_for_graph, create_app
from app.models import create_chat_model


def configured_settings(**overrides: object) -> Settings:
    defaults: dict[str, object] = {
        "llm_base_url": "https://example.invalid/v1",
        "llm_model": "test-model",
    }
    defaults.update(overrides)
    return Settings(_env_file=None, **defaults)


def user_message(text: str) -> dict[str, object]:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


class RecordingFakeChatModel(FakeListChatModel):
    """Fake streaming model that records the history sent to the model."""

    _seen_messages: list[list[BaseMessage]] = PrivateAttr(default_factory=list)

    async def _astream(self, messages, **kwargs):
        self._seen_messages.append(list(messages))
        async for chunk in super()._astream(messages, **kwargs):
            yield chunk


class CancellableFakeChatModel(FakeListChatModel):
    """Fake model for checking that a closed response cancels model streaming."""

    _was_cancelled: bool = PrivateAttr(default=False)

    async def _astream(self, messages, **kwargs):
        try:
            async for chunk in super()._astream(messages, **kwargs):
                yield chunk
        finally:
            self._was_cancelled = True


class StaticEventStream(httpx.AsyncByteStream):
    """Small OpenAI-compatible SSE response split across transport chunks."""

    def __init__(self, events: list[str]) -> None:
        self.events = events

    async def __aiter__(self):
        for event in self.events:
            yield event.encode()

    async def aclose(self) -> None:
        return None


def test_health_reports_configuration_without_disclosing_it() -> None:
    client = TestClient(create_app(settings=configured_settings()))

    response = client.get("/api/health")

    assert response.json() == {"status": "ok", "model_configured": True}
    assert "example.invalid" not in response.text


def test_missing_configuration_keeps_health_available_and_chat_actionable() -> None:
    client = TestClient(create_app(settings=Settings(_env_file=None)))

    assert client.get("/api/health").json() == {
        "status": "ok",
        "model_configured": False,
    }
    response = client.post("/api/chat", json={"messages": [user_message("Hi")]})

    assert response.status_code == 503
    assert "LLM_BASE_URL and LLM_MODEL" in response.json()["detail"]


def test_streams_incremental_text_and_uses_the_full_supplied_history() -> None:
    model = RecordingFakeChatModel(responses=["Hello from the model."])
    client = TestClient(create_app(settings=configured_settings(), model=model))

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                user_message("First turn"),
                {"role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
                user_message("Second turn"),
                {"role": "assistant", "content": []},
            ]
        },
    )

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    deltas = [json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")]
    assert "".join(deltas) == "Hello from the model."
    assert len(deltas) > 1
    history = model._seen_messages[0]
    assert [message.content for message in history] == [
        SYSTEM_PROMPT,
        "First turn",
        "Prior answer",
        "Second turn",
    ]


def test_empty_assistant_placeholder_is_removed_but_nonempty_history_is_kept() -> None:
    messages = _messages_for_graph(
        [
            user_message("Question"),
            {"role": "assistant", "content": [{"type": "text", "text": "Earlier reply"}]},
            {"role": "assistant", "content": []},
        ]
    )

    assert [message.content for message in messages] == ["Question", "Earlier reply"]


def test_conversations_are_independent_when_each_request_supplies_its_history() -> None:
    model = FakeListChatModel(responses=["First answer", "Other conversation"])
    client = TestClient(create_app(settings=configured_settings(), model=model))

    first = client.post("/api/chat", json={"messages": [user_message("A")]})
    second = client.post("/api/chat", json={"messages": [user_message("B")]})

    first_deltas = [
        json.loads(line[2:]) for line in first.text.splitlines() if line.startswith("0:")
    ]
    second_deltas = [
        json.loads(line[2:]) for line in second.text.splitlines() if line.startswith("0:")
    ]
    assert "".join(first_deltas) == "First answer"
    assert "".join(second_deltas) == "Other conversation"


def test_rejects_non_text_parts_and_unsupported_roles() -> None:
    client = TestClient(
        create_app(
            settings=configured_settings(),
            model=FakeListChatModel(responses=["unused"]),
        )
    )

    image = client.post(
        "/api/chat",
        json={"messages": [{"role": "user", "content": [{"type": "file", "data": "x"}]}]},
    )
    tool = client.post("/api/chat", json={"messages": [{"role": "tool", "content": "x"}]})

    assert image.status_code == 422
    assert tool.status_code == 422


def test_graph_is_built_from_an_injected_langchain_model() -> None:
    model = FakeListChatModel(responses=["A helpful answer"])
    graph = create_graph(model)

    response = asyncio.run(graph.ainvoke({"messages": [HumanMessage(content="A question")]}))

    assert response["messages"][-1].content == "A helpful answer"
    assert len(SYSTEM_PROMPT) > 0


def test_model_adapter_uses_only_the_current_endpoint_configuration(monkeypatch) -> None:
    calls: list[dict[str, object]] = []

    def fake_chat_openai(**kwargs: object) -> object:
        calls.append(kwargs)
        return object()

    monkeypatch.setattr("app.models.ChatOpenAI", fake_chat_openai)
    create_chat_model(
        configured_settings(
            llm_base_url="https://first.example/v1",
            llm_model="model-a",
            llm_api_key="first-key",
        )
    )
    create_chat_model(
        configured_settings(
            llm_base_url="https://second.example/v1",
            llm_model="model-b",
            llm_api_key="",
        )
    )

    assert calls[0]["base_url"] == "https://first.example/v1"
    assert calls[0]["model"] == "model-a"
    assert calls[0]["api_key"] == "first-key"
    assert calls[1]["base_url"] == "https://second.example/v1"
    assert calls[1]["model"] == "model-b"
    assert calls[1]["api_key"] == "not-required"
    assert calls[1]["use_responses_api"] is False
    assert calls[1]["stream_usage"] is False


def test_openai_compatible_stream_works_through_fastapi_and_langgraph() -> None:
    sent_requests: list[dict[str, object]] = []

    def event(delta: dict[str, str], finish_reason: str | None = None) -> str:
        payload = {
            "id": "test",
            "object": "chat.completion.chunk",
            "created": 0,
            "model": "test-model",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
        return f"data: {json.dumps(payload)}\n\n"

    events = [
        event({"role": "assistant"}),
        event({"content": "OpenAI-compatible "}),
        event({"content": "stream works."}),
        event({}, "stop"),
        "data: [DONE]\n\n",
    ]

    def respond(request: httpx.Request) -> httpx.Response:
        sent_requests.append(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            stream=StaticEventStream(events),
        )

    model_http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    model = ChatOpenAI(
        model="test-model",
        base_url="https://example.invalid/v1",
        api_key="test-key",
        use_responses_api=False,
        stream_usage=False,
        streaming=True,
        http_async_client=model_http_client,
    )
    client = TestClient(create_app(settings=configured_settings(), model=model))

    response = client.post(
        "/api/chat",
        json={
            "messages": [
                user_message("First turn"),
                {"role": "assistant", "content": [{"type": "text", "text": "Prior answer"}]},
                user_message("Second turn"),
                {"role": "assistant", "content": []},
            ]
        },
    )
    asyncio.run(model_http_client.aclose())

    deltas = [json.loads(line[2:]) for line in response.text.splitlines() if line.startswith("0:")]
    assert response.status_code == 200
    assert "".join(deltas) == "OpenAI-compatible stream works."
    assert len(deltas) > 1
    sent_messages = sent_requests[0]["messages"]
    assert [message["content"] for message in sent_messages] == [
        SYSTEM_PROMPT,
        "First turn",
        "Prior answer",
        "Second turn",
    ]


def test_model_failure_keeps_partial_text_and_reports_a_safe_actionable_error() -> None:
    model = FakeListChatModel(responses=["partial response"], error_on_chunk_number=7)
    client = TestClient(create_app(settings=configured_settings(), model=model))

    response = client.post("/api/chat", json={"messages": [user_message("Hi")]})

    assert response.status_code == 200
    assert '0:"p"' in response.text
    assert "model request failed" in response.text
    assert "example.invalid" not in response.text


def test_closing_a_stream_cancels_an_active_model_request() -> None:
    model = CancellableFakeChatModel(responses=["a long response"], sleep=0.01)
    application = create_app(settings=configured_settings(), model=model)
    endpoint = next(
        route.endpoint
        for route in application.routes
        if getattr(route, "path", None) == "/api/chat"
    )

    async def receive_one_chunk_and_close() -> bytes:
        response = await endpoint(ChatRequest(messages=[user_message("Stop this response")]))
        stream = response.body_iterator
        first_chunk = await anext(stream)
        await stream.aclose()
        return first_chunk

    first_chunk = asyncio.run(receive_one_chunk_and_close())

    assert "0:" in first_chunk
    assert model._was_cancelled
