from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class ParseRequest(BaseModel):
    message: str = Field(..., min_length=1)


class SearchRequest(BaseModel):
    criteria: dict[str, Any]
    top_k: int = Field(default=5, ge=1, le=20)


class RAGSearchRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] | None = None


class RAGRebuildRequest(BaseModel):
    force_rebuild: bool = True


class ProductResult(BaseModel):
    product_id: str
    product_name: str
    category_code: str | None = None
    category_name: str | None = None
    product_type: str | None = None
    brand: str | None = None
    tags: str | None = None
    short_description: str | None = None
    price_vnd: int | None = None
    sale_price_vnd: int | None = None
    effective_price_vnd: int | None = None
    stock_total: int | None = None
    sku: str | None = None
    size: str | None = None
    color: str | None = None
    stock: int | None = None
    variant_status: str | None = None
    rating: float | None = None
    sold_30d: int | None = None
    score: float | None = None
    matched_reasons: list[str] = Field(default_factory=list)


class ChatResponse(BaseModel):
    conversation_id: str = "default"
    pending_action: str | None = None
    criteria: dict[str, Any]
    reply: str
    products: list[ProductResult] = Field(default_factory=list)
    product_ids: list[str] = Field(default_factory=list)
    order_draft: dict[str, Any] | None = None
    rag_contexts: list[dict[str, Any]] = Field(default_factory=list)
    conversation_memory: dict[str, Any] = Field(default_factory=dict)
