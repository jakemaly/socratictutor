"""Small, provider-independent LangGraph chat graph."""

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, MessagesState, StateGraph

SYSTEM_PROMPT = "You are a helpful, clear assistant. Answer questions directly and accurately."


def create_graph(model: BaseChatModel):
    """Compile a single-node graph around any LangChain chat model."""

    async def chat(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = None
        async for chunk in model.astream(
            [SystemMessage(content=SYSTEM_PROMPT), *state["messages"]]
        ):
            response = chunk if response is None else response + chunk
        return {"messages": [response] if response is not None else []}

    graph = StateGraph(MessagesState)
    graph.add_node("chat", chat)
    graph.add_edge(START, "chat")
    graph.add_edge("chat", END)
    return graph.compile()
