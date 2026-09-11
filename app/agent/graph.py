from typing import Annotated, TypedDict

from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from app.agent.prompts import SYSTEM_PROMPT
from app.agent.tools import TOOLS
from app.config import settings
from app.llm import get_llm
import structlog
import time


log = structlog.get_logger()

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    iteration_count: int


_llm_with_tools = None


def _get_llm_with_tools():
    global _llm_with_tools
    if _llm_with_tools is None:
        _llm_with_tools = get_llm().bind_tools(TOOLS)
    return _llm_with_tools


_REPL_NUDGE = (
    "The user asked for a calculation via python_repl. You have not called "
    "python_repl yet. Call it NOW with real code (use print()). "
    "Do not describe the call in prose."
)


def _user_wants_repl(state: AgentState) -> bool:
    used_repl = False
    user = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            user = (msg.content or "").lower()
        if isinstance(msg, ToolMessage) and getattr(msg, "name", "") == "python_repl":
            used_repl = True
    if used_repl:
        return False
    return "python_repl" in user or any(
        key in user for key in ("compute", "calculate", "посчитай", "вычисли")
    )


def agent_node(state: AgentState) -> dict:
    t0 = time.perf_counter()
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    if _user_wants_repl(state):
        messages.append(SystemMessage(content=_REPL_NUDGE))
    response = _get_llm_with_tools().invoke(messages)
    latency_ms = int((time.perf_counter() - t0) * 1000)
    tool_calls = getattr(response, "tool_calls", None) or []
    log.info(
        "agent_node_completed",
        iteration=state["iteration_count"] + 1,
        latency_ms=latency_ms,
        tool_calls_count=len(tool_calls),
        tools_requested=[tc["name"] for tc in tool_calls],
    )
    return {
        "messages": [response],
        "iteration_count": state["iteration_count"] + 1,
    }


tool_executor_node = ToolNode(TOOLS)


def should_continue(state: AgentState) -> str:
    if state["iteration_count"] >= settings.max_iterations:
        return END
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", None):
        return "tool_executor"
    return END


def build_agent_graph(checkpointer=None):
    if checkpointer is None:
        checkpointer = MemorySaver()
    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tool_executor", tool_executor_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", should_continue)
    graph.add_edge("tool_executor", "agent")
    return graph.compile(checkpointer=checkpointer)