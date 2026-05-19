"""
Local LLM generation using Ollama.

This avoids OpenAI/Gemini API quota for answer generation.
Make sure Ollama is running locally before using this file.
"""

import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llama3.2:3b"


def build_rag_prompt(query: str, contexts: list[str]) -> str:
    """
    Build a grounded RAG prompt using retrieved documents.
    """
    context_text = "\n\n".join(
        f"[Document {i + 1}]\n{text}"
        for i, text in enumerate(contexts)
    )

    return f"""
You are a question-answering assistant.

Answer the question using only the provided context.
If the answer is not supported by the context, say:
"The retrieved context does not contain enough evidence."

Question:
{query}

Context:
{context_text}

Answer:
""".strip()


def generate_with_ollama(
    prompt: str,
    model: str = OLLAMA_MODEL,
    timeout: int = 180,
) -> str:
    """
    Generate an answer using a local Ollama model.
    """
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
        },
        timeout=timeout,
    )

    response.raise_for_status()
    return response.json()["response"]