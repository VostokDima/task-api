from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Только секрет — без default, контейнер падает если не задан.
    llm_api_key: str

    # LLM provider. Любой OpenAI-совместимый /chat/completions.
    # VseGPT вместо OpenRouter: последний отдаёт Cloudflare 403 на любой
    # запрос с RU-адреса (весь домен, до аутентификации), а VPS — российский.
    llm_base_url: str = "https://api.vsegpt.ru/v1"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    llm_temperature: float = 0.0

    # Прокси для LLM-запросов: "" — напрямую, "auto" — искать локальный
    # Clash/v2Ray на 10809, либо явный http://host:port.
    # Нужен только если base_url смотрит на заблокированный из RU домен.
    llm_proxy: str = ""

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "sklearn_docs"
    top_k: int = 4

    # Embeddings (e5 — мультиязычный, нужно для русского)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    normalize_embeddings: bool = True


settings = Settings()