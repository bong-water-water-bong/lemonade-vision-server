from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
import numpy as np


class VectorStore:
    def __init__(self, chroma_path: str | Path) -> None:
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._visual = self._client.get_or_create_collection("product_visual")
        self._text = self._client.get_or_create_collection("product_text")

    def upsert_visual(
        self, product_id: str, embedding: np.ndarray, metadata: dict[str, Any]
    ) -> None:
        self._visual.upsert(
            ids=[product_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

    def upsert_text(self, product_id: str, embedding: np.ndarray, metadata: dict[str, Any]) -> None:
        self._text.upsert(
            ids=[product_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

    def query_text(self, vector: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        count = self._text.count()
        if count == 0:
            return []
        n_results = min(top_k, count)
        results = self._text.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        return _query_rows(results)

    def query_visual(self, vector: np.ndarray, top_k: int = 3) -> list[dict[str, Any]]:
        count = self._visual.count()
        if count == 0:
            return []
        n_results = min(top_k, count)
        results = self._visual.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        return _query_rows(results)

    def product_count(self) -> int:
        return self._visual.count()


def _query_rows(results: Any) -> list[dict[str, Any]]:
    ids = results.get("ids") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    out: list[dict[str, Any]] = []
    for id_, meta, dist in zip(ids[0], metadatas[0], distances[0]):
        out.append({"id": id_, "metadata": meta, "distance": dist})
    return out
