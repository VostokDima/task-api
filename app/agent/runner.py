"""CLI smoke-test агента без HTTP-стека.

Usage:
    python -m app.agent.runner "What is Ridge regression?"
"""

from __future__ import annotations

import sys
import uuid

from langchain_core.messages import HumanMessage

from app.agent.graph import build_agent_graph
from app.agent.guardrails import GuardrailError, check_input, check_output


def run(question: str) -> None:
    try:
        check_input(question)
    except GuardrailError as exc:
        print(f"input rejected: {exc}")
        sys.exit(2)

    graph = build_agent_graph()
    result = graph.invoke(
        {"messages": [HumanMessage(content=question)], "iteration_count": 0},
        config={"configurable": {"thread_id": str(uuid.uuid4())}},
    )
    answer = result["messages"][-1].content or ""
    tools: list[str] = []
    for msg in result["messages"]:
        for tc in getattr(msg, "tool_calls", None) or []:
            tools.append(tc["name"])
    safe, reason = check_output(answer, tools)
    print(safe)
    print()
    print(f"iterations: {result['iteration_count']}")
    if tools:
        print(f"tools: {', '.join(tools)}")
    if reason:
        print(f"guardrail: {reason}")


def main() -> None:
    if len(sys.argv) < 2 or not sys.argv[1].strip():
        print('Usage: python -m app.agent.runner "<question>"', file=sys.stderr)
        sys.exit(1)
    run(sys.argv[1])


if __name__ == "__main__":
    main()
