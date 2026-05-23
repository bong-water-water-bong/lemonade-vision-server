from __future__ import annotations
from pathlib import Path
import numpy as np
import chromadb


class VectorStore:
    def __init__(self, chroma_path: str | Path) -> None:
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._visual = self._client.get_or_create_collection("product_visual")
        self._text = self._client.get_or_create_collection("product_text")

    def upsert_visual(self, product_id: str, embedding: np.ndarray, metadata: dict) -> None:
        self._visual.upsert(
            ids=[product_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

    def upsert_text(self, product_id: str, embedding: np.ndarray, metadata: dict) -> None:
        self._text.upsert(
            ids=[product_id],
            embeddings=[embedding.tolist()],
            metadatas=[metadata],
        )

    def query_text(self, vector: np.ndarray, top_k: int = 3) -> list[dict]:
        count = self._text.count()
        if count == 0:
            return []
        n_results = min(top_k, count)
        results = self._text.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        out = []
        for id_, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"id": id_, "metadata": meta, "distance": dist})
        return out

    def query_visual(self, vector: np.ndarray, top_k: int = 3) -> list[dict]:
        count = self._visual.count()
        if count == 0:
            return []
        n_results = min(top_k, count)
        results = self._visual.query(
            query_embeddings=[vector.tolist()],
            n_results=n_results,
            include=["metadatas", "distances"],
        )
        out = []
        for id_, meta, dist in zip(
            results["ids"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            out.append({"id": id_, "metadata": meta, "distance": dist})
        return out

    def product_count(self) -> int:
        return self._visual.count()
