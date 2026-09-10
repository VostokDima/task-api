from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session", autouse=True)
def mock_rag_chain():
    """Не поднимать e5/Qdrant/OpenRouter в pytest — lifespan только мок."""
    mock_chain = MagicMock()
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    with patch("app.main.build_rag_chain", return_value=(mock_chain, mock_retriever)):
        yield


@pytest.fixture(scope="session")
def client(mock_rag_chain):
    with TestClient(app) as c:
        yield c
