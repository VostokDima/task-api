# RAG-сервис по документации scikit-learn

Учебный ассистент Classic ML: retrieval по корпусу sklearn (linear models, trees, metrics) + LLM с цитатами `[1]`, `[2]`.

Раньше здесь был CRUD task-api — его больше нет.

**Прод:** https://92-242-60-220.nip.io/  
Gradio на `/`, API: `GET /health`, `POST /chat`, swagger `/docs`.

## Стек

FastAPI, Gradio, LangChain LCEL, Qdrant, `intfloat/multilingual-e5-small` (локальные эмбеддинги), любой OpenAI-совместимый `/chat/completions`.

На VPS с российским IP `openrouter.ai` отвечает Cloudflare 403 на весь домен (до ключа и модели). Прод ходит в [Hugging Face Inference Providers](https://huggingface.co/docs/inference-providers): `LLM_BASE_URL=https://router.huggingface.co/v1`. Локально через прокси можно оставить OpenRouter.

## Структура

```
app/
  main.py              FastAPI + lifespan RAG + Gradio (стрим, тайминги, источники)
  config.py            настройки из .env (LLM, Qdrant, эмбеддер)
  llm.py               urllib-клиент chat/completions, опциональный прокси
  rag/chain.py         retriever → prompt → LLM
  schemas/chat.py      ChatRequest / ChatResponse
  scripts/
    load_corpus.py     sklearn HTML + data/local/*.md → data/corpus_chunks.jsonl
    index_corpus.py    чанки → коллекция Qdrant sklearn_docs
    check_rag_repl.py  ручной прогон вопросов
data/local/about.md    «что умеет ассистент» (meta-вопросы)
notebooks/rag_eval.ipynb   оценка Ragas
tests/                 pytest (LLM мокнут); smoke инъекций — отдельно
docker-compose.yml     api + qdrant; caddy под profiles: [prod]
Caddyfile              HTTPS, Let's Encrypt, SSE без буфера
.github/workflows/     pytest + GHCR + деплой на VPS
```

`app/rag_code/` и `app/core/` — наследие, рантайм их не импортирует.

## API

`POST /chat`

```json
{"question": "What is Ridge regression?"}
```

Ответ: `answer` + `sources[]` (`url`, `snippet`). Пустой вопрос → 422, LLM недоступен → 503. Вопрос до 500 символов.

## Переменные окружения

| Переменная | Смысл | Пример прода |
|---|---|---|
| `LLM_API_KEY` | ключ шлюза, **обязателен** | `hf_...` |
| `LLM_BASE_URL` | OpenAI-compatible base | `https://router.huggingface.co/v1` |
| `LLM_MODEL` | slug модели шлюза | `meta-llama/Llama-3.3-70B-Instruct` |
| `LLM_PROXY` | `""` / `auto` / `http://host:port` | пусто (HF с RU доступен) |
| `QDRANT_URL` | векторное хранилище | `http://qdrant:6333` в Docker, `http://localhost:6333` локально |
| `RAG_DOMAIN` | хост Caddy | `92-242-60-220.nip.io` |

Остальное (top_k=4, e5-small, коллекция `sklearn_docs`) — дефолты в `app/config.py`. `.env` в git не коммитится.

## Локальный запуск

Нужны Python 3.11, `.env` с ключом, поднятый Qdrant с проиндексированной коллекцией.

```powershell
conda create -y -n task-api python=3.11
conda activate task-api
pip install -r requirements.txt
docker compose up -d qdrant
python -m app.scripts.load_corpus
python -m app.scripts.index_corpus
uvicorn app.main:app --reload
```

UI: http://localhost:8000/  
Docs: http://localhost:8000/docs

Индексацию не гонять одновременно с `api`: две копии e5+torch на слабой машине уходят в OOM. Сначала только `qdrant`, job `--rm`, потом `api`.

Весь стек в Docker (без Caddy):

```powershell
docker compose up -d --build api qdrant
```

## Прод (VPS)

```powershell
docker compose --profile prod up -d
```

Caddy слушает 80/443, выпускает сертификат на `RAG_DOMAIN`, проксирует на `api:8000`. Профиль нужен, чтобы локальный `compose up` не пытался пройти Let's Encrypt.

Первая индексация на сервере (swap ≥ 2G, `api` ещё не поднят):

```bash
docker compose up -d qdrant
docker compose run --rm api sh -c "python -m app.scripts.load_corpus && python -m app.scripts.index_corpus"
docker compose --profile prod up -d
```

## Тесты

```powershell
pytest tests/ -v
```

Цепочка RAG в `conftest.py` замокана — CI не ходит в LLM/Qdrant.

Live smoke prompt injection (нужен живой сервис):

```powershell
$env:RAG_SMOKE_URL = "https://92-242-60-220.nip.io"
pytest tests/test_prompt_injection_smoke.py -v -m smoke
```

Ragas: `notebooks/rag_eval.ipynb` (`ragas==0.2.15` запинен из-за сломанного импорта в 0.4.x).

## CI/CD

Push/PR в `main`: pytest → сборка `ghcr.io/<owner>/task-api` → scp `docker-compose.yml` + `Caddyfile` → на VPS пишется `.env` из GitHub Secrets и `docker compose --profile prod up -d`.

Секреты: `LLM_API_KEY`, `GHCR_TOKEN`, `VPS_HOST`, `SSH_PRIVATE_KEY`, `RAG_DOMAIN`.  
[Замечание] текущий workflow не прокидывает `LLM_BASE_URL` / `LLM_MODEL` — после деплоя с `main` они откатятся на дефолты `app/config.py` (сейчас VseGPT), и HF-прод отвалится, пока это не добавят в генерируемый `.env`.
