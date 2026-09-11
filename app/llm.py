import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils.function_calling import convert_to_openai_tool

from app.config import settings


def _proxy_reachable(url: str) -> bool:
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=0.6):
            return True
    except OSError:
        return False


def _proxy_url() -> str | None:
    configured = settings.llm_proxy.strip()
    if not configured:
        return None
    if configured != "auto":
        return configured
    # auto: локальный VPN-клиент. Автодетект осознанно НЕ включён по умолчанию —
    # иначе запрос к доступному напрямую шлюзу молча уедет через туннель.
    candidates: list[str] = []
    if Path("/.dockerenv").exists():
        candidates.append("http://host.docker.internal:10809")
    candidates.append("http://127.0.0.1:10809")
    for url in candidates:
        if _proxy_reachable(url):
            return url
    return None


def _opener() -> urllib.request.OpenerDirector:
    proxy = _proxy_url()
    if not proxy:
        return urllib.request.build_opener()
    return urllib.request.build_opener(
        urllib.request.ProxyHandler({"http": proxy, "https": proxy})
    )


def _chat_request(
    messages: list[BaseMessage],
    stop: list[str] | None,
    stream: bool,
    tools: list[dict] | None = None,
    tool_choice: str | dict | None = None,
) -> urllib.request.Request:
    body: dict = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "messages": [_message_to_openai(m) for m in messages],
        "stream": stream,
    }
    if stop:
        body["stop"] = stop
    if tools:
        body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
    return urllib.request.Request(
        f"{settings.llm_base_url.rstrip('/')}/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {settings.llm_api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:8000",
            "X-Title": "task-api",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) task-api-rag",
        },
        method="POST",
    )


def _chunk_from_openai_delta(delta: dict | None) -> ChatGenerationChunk | None:
    """Собрать AIMessageChunk из SSE-дельты, включая tool_calls без content."""
    if not delta:
        return None
    content = delta.get("content") or ""
    tool_call_chunks: list[dict] = []
    for tc in delta.get("tool_calls") or []:
        fn = tc.get("function") or {}
        tool_call_chunks.append(
            {
                "name": fn.get("name") or None,
                "args": fn.get("arguments") or "",
                "id": tc.get("id"),
                "index": tc.get("index", 0),
                "type": "tool_call_chunk",
            }
        )
    if not content and not tool_call_chunks:
        return None
    return ChatGenerationChunk(
        message=AIMessageChunk(content=content, tool_call_chunks=tool_call_chunks)
    )


def _message_to_chunk(message: AIMessage) -> AIMessageChunk:
    tool_call_chunks = [
        {
            "name": tc["name"] if isinstance(tc, dict) else tc.name,
            "args": json.dumps(
                tc["args"] if isinstance(tc, dict) else tc.args,
                ensure_ascii=False,
            ),
            "id": tc["id"] if isinstance(tc, dict) else tc.id,
            "index": i,
            "type": "tool_call_chunk",
        }
        for i, tc in enumerate(getattr(message, "tool_calls", None) or [])
    ]
    return AIMessageChunk(
        content=message.content or "",
        tool_call_chunks=tool_call_chunks,
    )


def _parse_tool_calls(raw: list | None) -> list[dict]:
    calls: list[dict] = []
    for item in raw or []:
        fn = item.get("function") or {}
        args = fn.get("arguments") or "{}"
        if isinstance(args, str):
            try:
                args = json.loads(args) if args.strip() else {}
            except json.JSONDecodeError:
                args = {"__raw": args}
        calls.append(
            {
                "name": fn.get("name") or "",
                "args": args if isinstance(args, dict) else {"value": args},
                "id": item.get("id") or "",
                "type": "tool_call",
            }
        )
    return calls


def _message_to_openai(message: BaseMessage) -> dict:
    if isinstance(message, ToolMessage) or message.type == "tool":
        return {
            "role": "tool",
            "content": str(message.content),
            "tool_call_id": getattr(message, "tool_call_id", "") or "",
        }
    role = message.type
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    elif role == "system":
        role = "system"
    else:
        role = "user"
    payload: dict = {"role": role, "content": str(message.content) if message.content is not None else ""}
    tool_calls = getattr(message, "tool_calls", None) or []
    if role == "assistant" and tool_calls:
        payload["tool_calls"] = [
            {
                "id": tc["id"] if isinstance(tc, dict) else tc.id,
                "type": "function",
                "function": {
                    "name": tc["name"] if isinstance(tc, dict) else tc.name,
                    "arguments": json.dumps(
                        tc["args"] if isinstance(tc, dict) else tc.args,
                        ensure_ascii=False,
                    ),
                },
            }
            for tc in tool_calls
        ]
        if not payload["content"]:
            payload["content"] = None
    return payload


class OpenAICompatChat(BaseChatModel):
    """Любой OpenAI-совместимый /chat/completions через urllib.
    Провайдер задаётся llm_base_url, туннель (если нужен) — llm_proxy."""

    @property
    def _llm_type(self) -> str:
        return "openai-compat-urllib"

    def bind_tools(self, tools: list, *, tool_choice: str | dict | None = "auto", **kwargs):
        formatted = [convert_to_openai_tool(t) for t in tools]
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        return self.bind(tools=formatted, **kwargs)

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        tools = kwargs.get("tools")
        tool_choice = kwargs.get("tool_choice")
        req = _chat_request(
            messages,
            stop,
            stream=False,
            tools=tools if isinstance(tools, list) else None,
            tool_choice=tool_choice if isinstance(tool_choice, (str, dict)) else None,
        )
        opener = _opener()
        last_error: Exception | None = None
        payload = None
        for _attempt in range(3):
            try:
                with opener.open(req, timeout=120) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                last_error = exc
        if payload is None:
            raise RuntimeError(f"LLM request failed: {last_error}") from last_error
        msg = payload["choices"][0]["message"]
        content = msg.get("content") or ""
        tool_calls = _parse_tool_calls(msg.get("tool_calls"))
        return ChatResult(
            generations=[
                ChatGeneration(message=AIMessage(content=content, tool_calls=tool_calls))
            ]
        )

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> Iterator[ChatGenerationChunk]:
        req = _chat_request(
            messages,
            stop,
            stream=True,
            tools=kwargs.get("tools") if isinstance(kwargs.get("tools"), list) else None,
            tool_choice=kwargs.get("tool_choice") if isinstance(kwargs.get("tool_choice"), (str, dict)) else None,
        )
        opener = _opener()
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                yielded = False
                with opener.open(req, timeout=120) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            break
                        payload = json.loads(data)
                        chunk = _chunk_from_openai_delta(
                            payload["choices"][0].get("delta")
                        )
                        if chunk is None:
                            continue
                        yielded = True
                        yield chunk
                if not yielded:
                    # tool_calls-only стрим без распарсенных дельт, либо пустой SSE
                    result = self._generate(
                        messages, stop=stop, run_manager=run_manager, **kwargs
                    )
                    yield ChatGenerationChunk(
                        message=_message_to_chunk(result.generations[0].message)
                    )
                return
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"LLM stream failed: {last_error}") from last_error


def get_llm() -> OpenAICompatChat:
    return OpenAICompatChat()
