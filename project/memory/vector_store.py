"""Cold-tier memory: FAISS index built from a persona's knowledge JSON.

Embeddings via local Ollama (`nomic-embed-text`) — runs once at index build.
Each persona gets its own namespaced index, cached in process memory.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import requests


OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")


def _embed(texts: list[str]) -> np.ndarray:
    """Call Ollama /api/embed for a batch of texts."""
    res = requests.post(
        f"{OLLAMA_BASE_URL}/api/embed",
        json={"model": EMBED_MODEL, "input": texts},
        timeout=60,
    )
    res.raise_for_status()
    embs = res.json()["embeddings"]
    arr = np.array(embs, dtype="float32")
    faiss.normalize_L2(arr)  # cosine via inner product
    return arr


@dataclass
class Document:
    id: str
    title: str
    content: str

    def as_text(self) -> str:
        return f"{self.title}\n{self.content}"


class PersonaIndex:
    def __init__(self, namespace: str, docs: list[Document], embeddings: np.ndarray):
        self.namespace = namespace
        self.docs = docs
        self.index = faiss.IndexFlatIP(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query: str, k: int = 3) -> list[Document]:
        docs, _ = self.search_with_score(query, k)
        return docs

    def search_with_score(
        self, query: str, k: int = 3
    ) -> tuple[list[Document], float]:
        """Return (docs, top_cosine). Vectors are L2-normalized so IP == cosine.

        top_cosine is the score of the best hit. Use it as a relevance gate
        to decide whether to inject retrieved context at all.
        """
        q = _embed([query])
        scores, idx = self.index.search(q, min(k, len(self.docs)))
        docs = [self.docs[i] for i in idx[0] if i != -1]
        top_score = float(scores[0][0]) if len(scores[0]) else 0.0
        return docs, top_score


class VectorStore:
    def __init__(self) -> None:
        self._indices: dict[str, PersonaIndex] = {}

    def load_or_build(self, knowledge_path: str, namespace: str) -> PersonaIndex:
        if namespace in self._indices:
            return self._indices[namespace]

        path = Path(knowledge_path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge file not found: {knowledge_path}")

        payload = json.loads(path.read_text())
        docs = [Document(**d) for d in payload.get("documents", [])]
        if not docs:
            raise ValueError(f"No documents in {knowledge_path}")

        embeddings = _embed([d.as_text() for d in docs])
        idx = PersonaIndex(namespace=namespace, docs=docs, embeddings=embeddings)
        self._indices[namespace] = idx
        return idx

    def get(self, namespace: str) -> Optional[PersonaIndex]:
        return self._indices.get(namespace)


vector_store = VectorStore()
