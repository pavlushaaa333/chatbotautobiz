from __future__ import annotations

import json
from typing import Any

import numpy as np

from app.normalizer import contains_phrase, normalize_text
from app.rag_embeddings import (
    DEFAULT_EMBEDDING_MODEL,
    FALLBACK_EMBEDDING_BACKEND,
    AutoBizEmbeddingModel,
)
from app.rag_indexer import (
    COLLECTION_NAME,
    DEFAULT_PERSIST_DIR,
    INDEX_FILE,
    STATS_FILE,
    _import_chromadb,
    _resolve_persist_dir,
)


CATALOG_SCOPE_QUERY_TERMS = [
    "bánh mì",
    "nước cam",
    "đồ ăn",
    "đồ uống",
    "trà sữa",
    "salad",
    "mỹ phẩm",
    "serum",
    "sữa rửa mặt",
    "mẹ và bé",
    "bỉm",
    "bình sữa",
    "giày sneaker",
    "giày dép",
    "quần áo đá bóng",
    "đồ đá bóng",
    "đồ thể thao chuyên dụng",
]


class AutoBizRAGRetriever:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.persist_path = _resolve_persist_dir(persist_dir)
        self.stats_path = self.persist_path / STATS_FILE
        self.local_index_path = self.persist_path / INDEX_FILE
        self.last_error: str | None = None
        self._embedding_model: AutoBizEmbeddingModel | None = None
        self._embedding_model_key: tuple[str, bool] | None = None

    def status(self) -> dict[str, Any]:
        stats = self._stats()
        ready = self._is_ready(stats)
        embedding_status = self._embedding_model.status() if self._embedding_model else None
        message = "RAG index ready" if ready else "RAG index not ready"
        if self.last_error:
            message = f"{message}: {self.last_error}"
        return {
            "ready": ready,
            "message": message,
            "persist_dir": str(self.persist_path),
            "backend": stats.get("backend"),
            "collection_name": stats.get("collection_name") or COLLECTION_NAME,
            "total_documents": stats.get("total_documents", 0),
            "embedding_backend": embedding_status.backend if embedding_status else stats.get("embedding_backend"),
            "embedding_model": stats.get("embedding_model"),
            "embedding_error": (
                embedding_status.error
                if embedding_status and embedding_status.error
                else stats.get("embedding_error")
            ),
        }

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_error = None
        stats = self._stats()
        if not self._is_ready(stats):
            self.last_error = "Vectorstore chưa được build. Chạy `python -m app.rag_indexer --rebuild`."
            return []

        effective_filters = filters
        if self._should_force_catalog_scope(query, filters):
            effective_filters = {"doc_type": "catalog_scope"}

        backend = stats.get("backend")
        if backend == "chroma":
            return self._retrieve_chroma(query, top_k=top_k, filters=effective_filters, stats=stats)
        return self._retrieve_local(query, top_k=top_k, filters=effective_filters, stats=stats)

    def _stats(self) -> dict[str, Any]:
        if not self.stats_path.exists():
            return {
                "backend": None,
                "total_documents": 0,
                "collection_name": COLLECTION_NAME,
            }
        try:
            return json.loads(self.stats_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            self.last_error = f"Không đọc được stats RAG: {exc}"
            return {"backend": None, "total_documents": 0, "collection_name": COLLECTION_NAME}

    def _is_ready(self, stats: dict[str, Any]) -> bool:
        if int(stats.get("total_documents") or 0) <= 0:
            return False
        if stats.get("backend") == "local_json":
            return self.local_index_path.exists()
        if stats.get("backend") == "chroma":
            chromadb = _import_chromadb()
            if chromadb is None:
                return False
            try:
                client = chromadb.PersistentClient(path=str(self.persist_path))
                collection = client.get_collection(stats.get("collection_name") or COLLECTION_NAME)
                return collection.count() > 0
            except Exception as exc:
                self.last_error = str(exc)
                return False
        return False

    def _embedding_model_for_stats(self, stats: dict[str, Any]) -> AutoBizEmbeddingModel:
        model_name = str(stats.get("embedding_model") or DEFAULT_EMBEDDING_MODEL)
        prefer_sentence_transformers = stats.get("embedding_backend") != FALLBACK_EMBEDDING_BACKEND
        key = (model_name, prefer_sentence_transformers)
        if self._embedding_model is None or self._embedding_model_key != key:
            self._embedding_model = AutoBizEmbeddingModel(
                model_name=model_name,
                prefer_sentence_transformers=prefer_sentence_transformers,
            )
            self._embedding_model_key = key
        return self._embedding_model

    def _retrieve_chroma(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        chromadb = _import_chromadb()
        if chromadb is None:
            self.last_error = "ChromaDB chưa được cài đặt."
            return []

        try:
            client = chromadb.PersistentClient(path=str(self.persist_path))
            collection = client.get_collection(stats.get("collection_name") or COLLECTION_NAME)
            embedding = self._embedding_model_for_stats(stats).encode([query])[0]
            results = collection.query(
                query_embeddings=[embedding],
                n_results=max(1, int(top_k)),
                where=self._chroma_where(filters),
                include=["documents", "metadatas", "distances"],
            )
            return self._normalize_chroma_results(results)
        except Exception as exc:
            self.last_error = str(exc)
            return []

    def _retrieve_local(
        self,
        query: str,
        *,
        top_k: int,
        filters: dict[str, Any] | None,
        stats: dict[str, Any],
    ) -> list[dict[str, Any]]:
        try:
            payload = json.loads(self.local_index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            self.last_error = f"Không đọc được local vector index: {exc}"
            return []

        documents = payload.get("documents") or []
        documents = [
            doc for doc in documents if self._metadata_matches(doc.get("metadata") or {}, filters)
        ]
        if not documents:
            return []

        query_embedding = np.asarray(
            self._embedding_model_for_stats(stats).encode([query])[0],
            dtype=np.float32,
        )
        scored: list[dict[str, Any]] = []
        for doc in documents:
            embedding = np.asarray(doc.get("embedding") or [], dtype=np.float32)
            if embedding.shape != query_embedding.shape:
                continue
            similarity = float(np.dot(query_embedding, embedding))
            distance = max(0.0, 1.0 - similarity)
            scored.append(
                {
                    "id": doc.get("id"),
                    "content": doc.get("content"),
                    "metadata": doc.get("metadata") or {},
                    "distance": distance,
                }
            )

        scored.sort(key=lambda item: item["distance"])
        return scored[: max(1, int(top_k))]

    def _normalize_chroma_results(self, results: dict[str, Any]) -> list[dict[str, Any]]:
        ids = (results.get("ids") or [[]])[0]
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        output: list[dict[str, Any]] = []
        for index, doc_id in enumerate(ids):
            output.append(
                {
                    "id": doc_id,
                    "content": documents[index] if index < len(documents) else "",
                    "metadata": metadatas[index] if index < len(metadatas) else {},
                    "distance": float(distances[index]) if index < len(distances) else None,
                }
            )
        return output

    def _metadata_matches(
        self,
        metadata: dict[str, Any],
        filters: dict[str, Any] | None,
    ) -> bool:
        if not filters:
            return True
        for key, expected in filters.items():
            if expected is None:
                continue
            if metadata.get(key) != expected:
                return False
        return True

    def _should_force_catalog_scope(
        self,
        query: str,
        filters: dict[str, Any] | None,
    ) -> bool:
        if filters:
            return False
        normalized_query = normalize_text(query)
        return any(contains_phrase(normalized_query, term) for term in CATALOG_SCOPE_QUERY_TERMS)

    def _chroma_where(self, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        clauses = [{key: value} for key, value in filters.items() if value is not None]
        if not clauses:
            return None
        if len(clauses) == 1:
            return clauses[0]
        return {"$and": clauses}
