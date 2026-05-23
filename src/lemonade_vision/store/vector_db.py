from __future__ import annotations
import numpy as np
import chromadb


class VectorStore:
    def __init__(self, chroma_path: str) -> None:
        self._client = chromadb.PersistentClient(path=chroma_path)
        self._visual = self._client.get_or_create_collection("product_visual")
        self._text = self._client.get_or_create_collection("product_text")

    def upsert_visual(self, id_: str, vector: np.ndarray, metadata: dict) -> None:
        self._visual.upsert(
            ids=[id_],
            embeddings=[vector.tolist()],
            metadatas=[metadata],
        )

    def upsert_text(self, id_: str, vector: np.ndarray, metadata: dict) -> None:
        self._text.upsert(
            ids=[id_],
            embeddings=[vector.tolist()],
            metadatas=[metadata],
        )

    def query_text(self, vector: np.ndarray, top_k: int = 3) -> list[dict]:
        results = self._text.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, self._text.count() or 1),
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
        results = self._visual.query(
            query_embeddings=[vector.tolist()],
            n_results=min(top_k, self._visual.count() or 1),
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
        return self._text.count()
