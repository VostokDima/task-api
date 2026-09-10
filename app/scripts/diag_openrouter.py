import json
import pathlib
import urllib.error
import urllib.request

text = pathlib.Path(".env").read_text(encoding="utf-8")
key = next(
    line.split("=", 1)[1].strip().strip('"')
    for line in text.splitlines()
    if line.split("=", 1)[0].strip() == "LLM_API_KEY"
)
url = "https://openrouter.ai/api/v1/chat/completions"
body = json.dumps(
    {
        "model": "meta-llama/llama-3.3-70b-instruct",
        "messages": [{"role": "user", "content": "Say hi in one word"}],
    }
).encode()
req = urllib.request.Request(
    url,
    data=body,
    headers={
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:8000",
        "X-Title": "task-api",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) task-api-rag",
    },
    method="POST",
)
opener = urllib.request.build_opener(
    urllib.request.ProxyHandler(
        {"http": "http://127.0.0.1:10809", "https": "http://127.0.0.1:10809"}
    )
)
try:
    with opener.open(req, timeout=30) as resp:
        print("OK", resp.status)
        print(resp.read()[:500].decode("utf-8", errors="replace"))
except urllib.error.HTTPError as exc:
    print("HTTP", exc.code)
    print("HEADERS", {k: exc.headers[k] for k in ("Server", "CF-RAY", "via") if exc.headers.get(k)})
    print("BODY", exc.read()[:400].decode("utf-8", errors="replace"))
except Exception as exc:
    print(type(exc).__name__, exc)
