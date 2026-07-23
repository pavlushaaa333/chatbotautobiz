from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from app.config import PROJECT_ROOT, product_data_source
from app.rag_documents import RAGDocument, build_rag_documents
from app.rag_embeddings import AutoBizEmbeddingModel


DEFAULT_PERSIST_DIR = "vectorstore/autobiz_fashion"
COLLECTION_NAME = "autobiz_fashion"
INDEX_FILE = "index.json"
STATS_FILE = "index_stats.json"


def _resolve_persist_dir(persist_dir: str | Path) -> Path:
    path = Path(persist_dir)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def _import_chromadb():
    try:
        import chromadb

        return chromadb
    except Exception:
        return None


def _doc_counts(documents: list[RAGDocument]) -> dict[str, int]:
    counts = Counter(str(doc.metadata.get("doc_type") or "unknown") for doc in documents)
    return {
        "total_documents": len(documents),
        "product_docs": counts.get("product", 0),
        "variant_docs": counts.get("variant", 0),
        "policy_docs": counts.get("policy", 0),
        "size_chart_docs": counts.get("size_chart", 0),
        "catalog_scope_docs": counts.get("catalog_scope", 0),
    }


class AutoBizRAGIndexer:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.persist_path = _resolve_persist_dir(persist_dir)
        self.stats_path = self.persist_path / STATS_FILE
        self.local_index_path = self.persist_path / INDEX_FILE

    def build_index(self, force_rebuild: bool = False) -> dict[str, Any]:
        if not force_rebuild and self._index_ready():
            stats = self.get_stats()
            stats["skipped"] = True
            return stats

        chromadb = _import_chromadb()
        cleanup_error = None
        if force_rebuild and self.persist_path.exists() and chromadb is not None:
            try:
                shutil.rmtree(self.persist_path)
            except PermissionError as exc:
                cleanup_error = str(exc)
        self.persist_path.mkdir(parents=True, exist_ok=True)

        documents = build_rag_documents()
        embedding_model = AutoBizEmbeddingModel()
        embeddings = embedding_model.encode(doc.content for doc in documents)
        embedding_status = embedding_model.status()

        if chromadb is not None:
            backend = "chroma"
            self._write_chroma_index(chromadb, documents, embeddings)
        else:
            backend = "local_json"
            self._write_local_index(documents, embeddings)

        stats = {
            **_doc_counts(documents),
            "product_data_source": product_data_source(),
            "persist_dir": str(self.persist_path),
            "backend": backend,
            "collection_name": COLLECTION_NAME,
            "embedding_backend": embedding_status.backend,
            "embedding_model": embedding_status.model_name,
            "embedding_dimension": embedding_status.dimension,
            "embedding_error": embedding_status.error,
            "cleanup_error": cleanup_error,
            "skipped": False,
        }
        self._write_stats(stats)
        return stats

    def get_stats(self) -> dict[str, Any]:
        if self.stats_path.exists():
            return json.loads(self.stats_path.read_text(encoding="utf-8"))
        return {
            "total_documents": 0,
            "product_docs": 0,
            "variant_docs": 0,
            "policy_docs": 0,
            "size_chart_docs": 0,
            "catalog_scope_docs": 0,
            "persist_dir": str(self.persist_path),
            "backend": None,
            "collection_name": COLLECTION_NAME,
            "embedding_backend": None,
            "embedding_model": None,
            "embedding_dimension": None,
            "embedding_error": None,
            "skipped": False,
        }

    def _index_ready(self) -> bool:
        stats = self.get_stats()
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
                collection = client.get_collection(COLLECTION_NAME)
                return collection.count() > 0
            except Exception:
                return False
        return self.local_index_path.exists()

    def _write_chroma_index(
        self,
        chromadb: Any,
        documents: list[RAGDocument],
        embeddings: list[list[float]],
    ) -> None:
        client = chromadb.PersistentClient(path=str(self.persist_path))
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 128
        for start in range(0, len(documents), batch_size):
            batch_docs = documents[start : start + batch_size]
            batch_embeddings = embeddings[start : start + batch_size]
            collection.add(
                ids=[doc.id for doc in batch_docs],
                documents=[doc.content for doc in batch_docs],
                metadatas=[doc.metadata for doc in batch_docs],
                embeddings=batch_embeddings,
            )

    def _write_local_index(
        self,
        documents: list[RAGDocument],
        embeddings: list[list[float]],
    ) -> None:
        payload = {
            "collection_name": COLLECTION_NAME,
            "documents": [
                {
                    "id": doc.id,
                    "content": doc.content,
                    "metadata": doc.metadata,
                    "embedding": embeddings[index],
                }
                for index, doc in enumerate(documents)
            ],
        }
        self.local_index_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _write_stats(self, stats: dict[str, Any]) -> None:
        self.stats_path.write_text(
            json.dumps(stats, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the AutoBiz Fashion RAG index.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and rebuild the existing index.")
    parser.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    args = parser.parse_args()

    stats = AutoBizRAGIndexer(args.persist_dir).build_index(force_rebuild=args.rebuild)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
