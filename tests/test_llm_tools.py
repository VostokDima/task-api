import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.llm import OpenAICompatChat, _message_to_openai, _parse_tool_calls


def test_bind_tools_does_not_raise():
    bound = OpenAICompatChat().bind_tools(
        [
            {
                "type": "function",
                "function": {
                    "name": "ping",
                    "description": "ping",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]
    )
    assert bound is not None


def test_parse_tool_calls_from_openai_payload():
    calls = _parse_tool_calls(
        [
            {
                "id": "call_1",
                "type": "function",
                "function": {"name": "python_repl", "arguments": '{"code": "print(1)"}'},
            }
        ]
    )
    assert calls[0]["name"] == "python_repl"
    assert calls[0]["args"] == {"code": "print(1)"}
    assert calls[0]["id"] == "call_1"


def test_message_to_openai_roundtrip_tool_call():
    assistant = AIMessage(
        content="",
        tool_calls=[{"name": "web_search", "args": {"query": "x"}, "id": "c1", "type": "tool_call"}],
    )
    payload = _message_to_openai(assistant)
    assert payload["role"] == "assistant"
    assert payload["tool_calls"][0]["function"]["name"] == "web_search"
    assert json.loads(payload["tool_calls"][0]["function"]["arguments"]) == {"query": "x"}

    tool = ToolMessage(content="ok", tool_call_id="c1")
    assert _message_to_openai(tool) == {"role": "tool", "content": "ok", "tool_call_id": "c1"}

    human = _message_to_openai(HumanMessage(content="Hi"))
    assert human == {"role": "user", "content": "Hi"}


def test_stream_delta_with_tool_calls_no_content():
    from langchain_core.language_models.chat_models import generate_from_stream

    from app.llm import _chunk_from_openai_delta

    first = _chunk_from_openai_delta(
        {
            "content": None,
            "tool_calls": [
                {
                    "index": 0,
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "python_repl", "arguments": ""},
                }
            ],
        }
    )
    second = _chunk_from_openai_delta(
        {
            "tool_calls": [
                {
                    "index": 0,
                    "function": {"arguments": '{"code": "print(10)"}'},
                }
            ]
        }
    )
    assert first is not None
    assert second is not None
    result = generate_from_stream(iter([first, second]))
    msg = result.generations[0].message
    assert msg.tool_calls[0]["name"] == "python_repl"
    assert msg.tool_calls[0]["args"] == {"code": "print(10)"}


def test_stream_delta_skips_empty_keepalive():
    from app.llm import _chunk_from_openai_delta

    assert _chunk_from_openai_delta({}) is None
    assert _chunk_from_openai_delta({"content": None}) is None
