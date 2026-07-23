from __future__ import annotations

from typing import Any

from app.config import product_data_source
from app.rag_indexer import AutoBizRAGIndexer, DEFAULT_PERSIST_DIR
from app.rag_retriever import AutoBizRAGRetriever


class AutoBizRAGService:
    def __init__(self, persist_dir: str = DEFAULT_PERSIST_DIR):
        self.persist_dir = persist_dir
        self.indexer = AutoBizRAGIndexer(persist_dir=persist_dir)
        self.retriever = AutoBizRAGRetriever(persist_dir=persist_dir)

    def status(self) -> dict[str, Any]:
        return self.retriever.status()

    def rebuild_index(self, force_rebuild: bool = True) -> dict[str, Any]:
        stats = self.indexer.build_index(force_rebuild=force_rebuild)
        status = self.retriever.status()
        return {**stats, "ready": status.get("ready"), "message": status.get("message")}

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        results = self.retriever.retrieve(query=query, top_k=top_k, filters=filters)
        if product_data_source() != "postgres":
            return results
        return [
            result
            for result in results
            if (result.get("metadata") or {}).get("doc_type") not in {"product", "variant"}
        ]

    def retrieve_contexts(
        self,
        query: str,
        criteria: dict[str, Any],
        products: list[dict[str, Any]] | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        contexts: list[dict[str, Any]] = []
        products = products or []

        intent = criteria.get("intent")
        if intent == "out_of_scope_request" or criteria.get("catalog_coverage") in {
            "not_supported",
            "unsupported",
        }:
            contexts.extend(
                self.search(query, top_k=max(1, top_k), filters={"doc_type": "catalog_scope"})
            )
            return self._format_contexts(contexts)

        if intent == "size_recommendation":
            filters: dict[str, Any] = {"doc_type": "size_chart"}
            if criteria.get("product_type"):
                filters["product_type"] = criteria["product_type"]
            if criteria.get("gender"):
                filters["gender"] = criteria["gender"]
            contexts.extend(self.search(query, top_k=max(top_k, 5), filters=filters))
            if not contexts and criteria.get("product_type"):
                contexts.extend(
                    self.search(
                        query,
                        top_k=max(top_k, 5),
                        filters={"doc_type": "size_chart", "product_type": criteria["product_type"]},
                    )
                )
            contexts.extend(
                self.search(
                    "hướng dẫn đo cơ thể đổi size form áo AutoBiz Fashion",
                    top_k=2,
                    filters={"doc_type": "policy"},
                )
            )
            return self._format_contexts(contexts)

        if intent in {"product_availability_check", "mixed_product_request"}:
            product_ids = self._availability_product_ids(criteria, products)
            for product_id in product_ids:
                contexts.extend(
                    self.search(query, top_k=top_k, filters={"product_id": product_id})
                )
            contexts.extend(
                self.search(
                    "COD freeship đổi trả tư vấn size AutoBiz Fashion",
                    top_k=2,
                    filters={"doc_type": "policy"},
                )
            )
            if intent == "mixed_product_request" or criteria.get("out_of_scope_items"):
                contexts.extend(
                    self.search(query, top_k=1, filters={"doc_type": "catalog_scope"})
                )
            if not contexts:
                contexts.extend(self.search(query, top_k=top_k))
            return self._format_contexts(contexts)

        product_type = criteria.get("product_type")
        if product_type:
            contexts.extend(
                self.search(query, top_k=max(top_k, 5), filters={"product_type": product_type})
            )
        if criteria.get("category_code"):
            contexts.extend(
                self.search(query, top_k=top_k, filters={"category_code": criteria["category_code"]})
            )
        if not contexts:
            contexts.extend(self.search(query, top_k=top_k))
        return self._format_contexts(contexts)

    def _availability_product_ids(
        self,
        criteria: dict[str, Any],
        products: list[dict[str, Any]],
    ) -> list[str]:
        product_ids = [str(product.get("product_id")) for product in products if product.get("product_id")]
        availability = criteria.get("availability") or {}
        product = availability.get("product") or {}
        if product.get("product_id"):
            product_ids.append(str(product["product_id"]))
        for alternative in availability.get("alternatives") or []:
            if alternative.get("product_id"):
                product_ids.append(str(alternative["product_id"]))
        return list(dict.fromkeys(product_ids))

    def _format_contexts(self, contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        formatted: list[dict[str, Any]] = []
        seen: set[str] = set()
        for context in contexts:
            doc_id = str(context.get("id") or "")
            if not doc_id or doc_id in seen:
                continue
            seen.add(doc_id)
            metadata = context.get("metadata") or {}
            if (
                product_data_source() == "postgres"
                and metadata.get("doc_type") in {"product", "variant"}
            ):
                continue
            distance = context.get("distance")
            score = None
            if distance is not None:
                try:
                    score = round(max(0.0, 1.0 - float(distance)), 4)
                except (TypeError, ValueError):
                    score = None
            content = str(context.get("content") or "")
            formatted.append(
                {
                    "id": doc_id,
                    "doc_type": metadata.get("doc_type"),
                    "product_id": metadata.get("product_id"),
                    "product_type": metadata.get("product_type"),
                    "gender": metadata.get("gender"),
                    "size": metadata.get("size"),
                    "sku": metadata.get("sku"),
                    "source": metadata.get("source") or doc_id,
                    "score": score,
                    "distance": distance,
                    "content_preview": self._preview(content),
                }
            )
        return formatted

    def _preview(self, content: str, limit: int = 180) -> str:
        compact = " ".join(str(content or "").split())
        if len(compact) <= limit:
            return compact
        return compact[: limit - 3].rstrip() + "..."
