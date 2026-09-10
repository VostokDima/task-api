import json
import socket
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

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


def _chat_request(messages: list[BaseMessage], stop: list[str] | None, stream: bool) -> urllib.request.Request:
    body: dict = {
        "model": settings.llm_model,
        "temperature": settings.llm_temperature,
        "messages": [_message_to_openai(m) for m in messages],
        "stream": stream,
    }
    if stop:
        body["stop"] = stop
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


def _message_to_openai(message: BaseMessage) -> dict[str, str]:
    role = message.type
    if role == "human":
        role = "user"
    elif role == "ai":
        role = "assistant"
    elif role == "system":
        role = "system"
    else:
        role = "user"
    return {"role": role, "content": str(message.content)}


class OpenAICompatChat(BaseChatModel):
    """Любой OpenAI-совместимый /chat/completions через urllib.
    Провайдер задаётся llm_base_url, туннель (если нужен) — llm_proxy."""

    @property
    def _llm_type(self) -> str:
        return "openai-compat-urllib"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> ChatResult:
        req = _chat_request(messages, stop, stream=False)
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
        text = payload["choices"][0]["message"]["content"]
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: object = None,
        **kwargs: object,
    ) -> Iterator[ChatGenerationChunk]:
        req = _chat_request(messages, stop, stream=True)
        opener = _opener()
        last_error: Exception | None = None
        for _attempt in range(3):
            try:
                with opener.open(req, timeout=120) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line.startswith("data:"):
                            continue
                        data = line[5:].strip()
                        if data == "[DONE]":
                            return
                        payload = json.loads(data)
                        delta = payload["choices"][0].get("delta", {}).get("content") or ""
                        if not delta:
                            continue
                        yield ChatGenerationChunk(message=AIMessageChunk(content=delta))
                return
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc
            except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError) as exc:
                last_error = exc
        raise RuntimeError(f"LLM stream failed: {last_error}") from last_error


def get_llm() -> OpenAICompatChat:
    return OpenAICompatChat()
