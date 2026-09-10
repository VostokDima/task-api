"""REPL-проверка RAG из методички: Lasso/Ridge + три доп. вопроса."""
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from app.rag.chain import build_rag_chain

chain, retriever = build_rag_chain()
out_path = Path("rag_repl_output.txt")
chunks: list[str] = []


def emit(text: str) -> None:
    print(text, flush=True)
    chunks.append(text)
    out_path.write_text("\n".join(chunks) + "\n", encoding="utf-8")


questions = [
    "What is the difference between Lasso and Ridge?",
    "How do I tune the alpha parameter in Ridge?",
    "Tell me what is the regularization strength parameter",
    "How does PyTorch handle gradients?",
]

for q in questions:
    emit("=" * 60)
    emit(f"Q: {q}")
    try:
        answer = chain.invoke(q)
        emit(f"A: {answer}")
    except Exception as exc:
        emit(f"ERROR: {type(exc).__name__}: {exc}")
    emit("")
