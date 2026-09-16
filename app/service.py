from __future__ import annotations

import re
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

import app.order_draft_service as order_draft_service
from app.config import active_shop_id, catalog_source, product_data_source
from app.data_loader import DataStore, load_data
from app.normalizer import normalize_text, strip_accents
from app.order_draft_service import (
    ORDER_PENDING_ACTIONS,
    ORDER_STATE_AWAITING_CONFIRMATION,
    ORDER_STATE_COLLECTING,
    ORDER_STATE_PENDING_SHOP_APPROVAL,
    ORDER_STATE_SUBMISSION_FAILED,
    ORDER_STATE_SUBMITTING,
    active_order_draft,
    build_summary,
    clear_item_variant,
    ensure_order_draft,
    extract_customer_bundle,
    extract_order_color_size,
    extract_phone_candidate,
    extract_quantity,
    first_item,
    is_cancel,
    is_order_confirmation_message,
    is_retry,
    is_valid_phone,
    looks_like_browsing_shift,
    missing_order_action,
    normalize_payment_method,
    normalize_phone,
    product_reference_index,
    recalculate_totals,
    requested_current_product,
    set_item_product,
    set_item_variant,
    submit_draft_order_to_n8n,
    text_key,
    validate_draft_order,
)
from app.parser import parse_customer_message
from app.product_repository import SAFE_PRODUCT_DATA_ERROR_REPLY
from app.rag_service import AutoBizRAGService
from app.response_generator import build_model_context, build_reply
from app.search_engine import (
    PostgresProductSearchEngine,
    ProductSearchEngine,
    resolve_product_price,
)
from app.size_recommender import recommend_size

CONVERSATION_STORE: dict[str, dict[str, Any]] = {}

AFFIRMATIVE_FOLLOWUPS = {
    "có",
    "có nhé",
    "ok",
    "oke",
    "được",
    "được nhé",
    "gợi ý đi",
    "xem thử",
    "cho mình xem",
    "cho tôi xem",
    "xem cũng được",
    "uh",
    "uhm",
    "ừ",
    "ừm",
    "được ạ",
    "có ạ",
}
NEGATIVE_FOLLOWUPS = {
    "không",
    "không nhé",
    "không ạ",
    "thôi",
    "thôi khỏi",
    "khỏi",
    "chưa",
    "để sau",
}
ALTERNATIVE_PRODUCT_INTENTS = {
    "alternative_product",
    "similar_product",
    "reject_current_product",
    "request_alternative_product",
    "request_similar_product",
}
ALTERNATIVE_PRODUCT_RESPONSE_MODES = {
    "alternative_product",
    "similar_product",
    "reject_current_product",
}
PREFERENCE_FIELDS = ["gender", "use_case", "color", "budget_max_vnd", "size"]
REQUIRED_COLLECT_FIELDS = ["gender", "use_case", "budget_max_vnd"]
PRICE_FILTER_DIRECTIONS = {"above", "below", "at_least", "between"}
GENERIC_QUESTION_PRODUCT_GROUPS = {
    "nào",
    "cái nào",
    "mẫu nào",
    "sản phẩm nào",
    "loại nào",
    "món nào",
    "có cái nào",
    "có mẫu nào",
}
MEMORY_DEFAULTS: dict[str, Any] = {
    "recommended_size": None,
    "alternative_size": None,
    "fit_preference": None,
    "height_cm": None,
    "weight_kg": None,
    "selected_product_id": None,
    "selected_variant_sku": None,
    "rejected_product_ids": [],
}
NEAREST_OVER_BUDGET_MULTIPLIER = 1.5
POLICY_LABELS = {
    "cod": "COD",
    "free_shipping": "freeship",
    "size_exchange": "đổi size",
    "inspection": "kiểm hàng",
    "delivery_time": "thời gian giao hàng",
}


class ChatbotService:
    def __init__(self, data_store: DataStore | None = None):
        self.catalog_source = catalog_source()
        self.product_data_source = product_data_source()
        self.active_shop_id = active_shop_id()
        self.data_store = data_store or load_data()
        if not self.data_store.products.empty:
            self.search_engine = ProductSearchEngine(
                self.data_store.products, self.data_store.variants
            )
        else:
            self.search_engine = PostgresProductSearchEngine()
        self.rag_service = AutoBizRAGService()

    def parse(self, message: str) -> dict[str, Any]:
        criteria = parse_customer_message(message, self.data_store.synonyms)
        criteria = self.search_engine.catalog_gate.apply(criteria)
        if self.active_shop_id:
            criteria["shop_id"] = self.active_shop_id
        criteria["catalog_source"] = self.catalog_source
        return criteria

    def search_products(
        self, criteria: dict[str, Any], top_k: int = 5
    ) -> list[dict[str, Any]]:
        products = self.search_engine.search(criteria, top_k=top_k)
        last_error = getattr(self.search_engine, "last_error", None)
        if last_error:
            criteria["product_data_error"] = last_error
            criteria["response_mode"] = "product_data_error"
        return products

    def _is_catalog_browsing(self, criteria: dict[str, Any]) -> bool:
        return criteria.get("response_mode") == "catalog_browsing"

    def _should_search_products(self, criteria: dict[str, Any]) -> bool:
        return bool(
            criteria.get("category_code")
            or criteria.get("product_type")
            or self._is_catalog_browsing(criteria)
            or criteria.get("budget_min_vnd") is not None
            or criteria.get("budget_max_vnd") is not None
        )

    def _classify_empty_product_search(
        self,
        criteria: dict[str, Any],
        products: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if products or not criteria.get("product_type") or criteria.get("product_data_error"):
            return products
        if criteria.get("catalog_coverage") in {"not_supported", "unsupported"}:
            return products

        exact_type_lookup = getattr(self.search_engine, "products_by_exact_type", None)
        if not callable(exact_type_lookup):
            return products

        exact_type_products = exact_type_lookup(criteria.get("product_type"))
        last_error = getattr(self.search_engine, "last_error", None)
        if last_error:
            criteria["product_data_error"] = last_error
            criteria["response_mode"] = "product_data_error"
            return products

        if not exact_type_products:
            criteria["response_mode"] = "product_type_not_found"
            criteria.pop("out_of_stock_products", None)
            return products

        in_stock_products = [
            product
            for product in exact_type_products
            if self._product_is_available_for_empty_search(product)
        ]
        if not in_stock_products:
            criteria["response_mode"] = "product_type_out_of_stock"
            criteria["out_of_stock_products"] = exact_type_products
            return products

        criteria["response_mode"] = "product_search_no_match"
        criteria.pop("out_of_stock_products", None)
        return products

    def search_rag(
        self,
        query: str,
        top_k: int = 5,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return self.rag_service.search(query, top_k=top_k, filters=filters)

    def rebuild_rag_index(self, force_rebuild: bool = True) -> dict[str, Any]:
        return self.rag_service.rebuild_index(force_rebuild=force_rebuild)

    def rag_status(self) -> dict[str, Any]:
        return self.rag_service.status()

    def chat(
        self, message: str, top_k: int = 5, conversation_id: str | None = None
    ) -> dict[str, Any]:
        conversation_id = conversation_id or "default"
        order_result = self._handle_order_message(message, conversation_id)
        if order_result is not None:
            return order_result

        followup_result = self._handle_pending_followup(message, top_k, conversation_id)
        if followup_result is not None:
            return followup_result

        criteria = self.parse(message)
        criteria = self._prepare_contextual_criteria(message, conversation_id, criteria)

        if criteria.get("intent") == "policy_question":
            return self._handle_policy_question(conversation_id, criteria)

        if criteria.get("intent") == "negative_feedback":
            return self._handle_negative_feedback(conversation_id, criteria)

        if criteria.get("response_mode") == "offer_closest_product_type":
            return self._handle_closest_product_type_offer(conversation_id, criteria)

        if criteria.get("intent") in {
            "request_alternative_product",
            "request_similar_product",
        }:
            return self._handle_alternative_product(
                message, top_k, conversation_id, criteria
            )
        if criteria.get("intent") == "reject_current_product":
            return self._handle_alternative_product(
                message, top_k, conversation_id, criteria
            )

        if criteria.get("intent") in {
            "variant_availability_check",
            "product_availability_check",
            "mixed_product_request",
        }:
            return self._handle_variant_availability(
                message, top_k, conversation_id, criteria
            )

        if criteria.get("intent") == "general_buying_intent":
            criteria["rag_contexts"] = []
            criteria["rag_status"] = self.rag_service.status()
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            reply = build_reply(criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=reply,
                products=[],
                rag_contexts=[],
            )

        if criteria.get("intent") == "size_recommendation":
            return self._handle_size_recommendation(
                message, top_k, conversation_id, criteria
            )

        if self._should_start_preference_collection(criteria):
            criteria = self._prepare_collect_preferences_criteria(
                criteria, conversation_id
            )
            criteria["rag_contexts"] = []
            criteria["rag_status"] = self.rag_service.status()
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        if criteria.get("intent") in {
            "out_of_scope_request",
        }:
            products = self._handle_structured_availability(criteria)
            rag_contexts = self._attach_rag_contexts(message, criteria, products, top_k)
            reply = build_reply(criteria, products)
            pending_action = self._store_pending_action(
                conversation_id, criteria, products
            )
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=reply,
                products=products,
                rag_contexts=rag_contexts,
            )

        rag_contexts = self._attach_rag_contexts(message, criteria, [], top_k)
        products: list[dict[str, Any]] = []
        if criteria.get("catalog_coverage") not in {
            "not_supported",
            "unsupported",
        } and self._should_search_products(criteria):
            if self._is_catalog_browsing(criteria):
                criteria["need_clarification"] = False
                criteria["clarification_questions"] = []
            products = self.search_products(criteria, top_k=top_k)
            products = self._rerank_products_with_rag(products, rag_contexts)
            if not products:
                products = self._attach_nearest_over_budget(criteria)
            products = self._classify_empty_product_search(criteria, products)

        reply = build_reply(criteria, products)
        pending_action = self._store_pending_action(conversation_id, criteria, products)
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=products,
            rag_contexts=rag_contexts,
        )

    def _handle_order_message(
        self,
        message: str,
        conversation_id: str,
    ) -> dict[str, Any] | None:
        context = dict(CONVERSATION_STORE.get(conversation_id) or {})
        pending_action = context.get("pending_action")
        draft = active_order_draft(context)
        criteria = self.parse(message)
        is_order_related = (
            pending_action in ORDER_PENDING_ACTIONS
            or draft is not None
            or criteria.get("intent") == "purchase_intent"
            or criteria.get("purchase_intent") is True
            or is_order_confirmation_message(message)
            or is_retry(message)
        )
        if not is_order_related:
            return None

        for field, default_value in MEMORY_DEFAULTS.items():
            if field not in context:
                context[field] = (
                    list(default_value)
                    if isinstance(default_value, list)
                    else default_value
                )

        if is_cancel(message):
            return self._cancel_order_draft(conversation_id, context, message)

        if draft is None and (
            is_order_confirmation_message(message) or is_retry(message)
        ):
            return self._order_response(
                conversation_id,
                context,
                None,
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "order_draft_missing_for_confirmation",
                },
                "Mình chưa có đơn nháp nào đang chờ xác nhận. Bạn cho mình biết sản phẩm muốn mua trước nhé.",
            )

        if (
            draft
            and draft.get("status") == ORDER_STATE_PENDING_SHOP_APPROVAL
            and (is_order_confirmation_message(message) or is_retry(message))
        ):
            return self._order_response(
                conversation_id,
                context,
                None,
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "order_already_submitted",
                },
                "Đơn của bạn đã được gửi sang shop và đang chờ duyệt rồi nhé.",
            )

        if (
            draft
            and draft.get("status") == ORDER_STATE_SUBMITTING
            and (is_order_confirmation_message(message) or is_retry(message))
        ):
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "order_submission_in_progress",
                },
                "Mình đang gửi đơn sang shop, bạn chờ mình một chút nhé.",
            )

        if pending_action in ORDER_PENDING_ACTIONS or draft is not None:
            if (
                criteria.get("intent") == "policy_question"
                and pending_action != "choose_payment_method"
            ):
                return self._answer_policy_during_order(
                    conversation_id,
                    context,
                    criteria,
                    message,
                )
            if self._looks_like_order_stock_question(message, context):
                return self._answer_stock_during_order(
                    conversation_id, context, message
                )
            if (
                pending_action
                in {
                    "collect_customer_name",
                    "collect_customer_phone",
                    "collect_customer_address",
                    "choose_payment_method",
                    "confirm_order_draft",
                }
                and looks_like_browsing_shift(message)
                and criteria.get("intent") != "purchase_intent"
            ):
                return self._order_response(
                    conversation_id,
                    context,
                    pending_action,
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "confirm_cancel_for_browsing_shift",
                    },
                    "Bạn có muốn hủy đơn nháp hiện tại để xem mẫu khác không ạ?",
                )

        if pending_action == "confirm_order_draft":
            if draft and draft.get("status") == ORDER_STATE_PENDING_SHOP_APPROVAL:
                return self._order_response(
                    conversation_id,
                    context,
                    None,
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "order_already_submitted",
                    },
                    "Đơn của bạn đã được gửi sang shop và đang chờ duyệt rồi nhé.",
                )
            if draft and draft.get("status") == ORDER_STATE_SUBMITTING:
                return self._order_response(
                    conversation_id,
                    context,
                    "confirm_order_draft",
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "order_submission_in_progress",
                    },
                    "Mình đang gửi đơn sang shop, bạn chờ mình một chút nhé.",
                )
            if is_order_confirmation_message(message) or (
                is_retry(message) and draft and draft.get("last_submission_failed")
            ):
                return self._submit_order_draft(conversation_id, context, message)
            edit_result = self._handle_order_edit(
                message, conversation_id, context, criteria
            )
            if edit_result is not None:
                return edit_result
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "confirm_order_draft",
                },
                "Dạ, nếu thông tin đã đúng bạn nhắn “Xác nhận”. Nếu cần sửa, bạn có thể nhắn nội dung muốn đổi giúp mình nhé.",
            )

        if pending_action == "choose_order_product":
            product = self._resolve_order_product(message, criteria, context)
            if product is None:
                return self._ask_order_product(
                    conversation_id, context, criteria, message
                )
            return self._start_or_update_order_product(
                conversation_id,
                context,
                criteria,
                message,
                product,
            )

        if draft is None and (
            criteria.get("intent") == "purchase_intent"
            or criteria.get("purchase_intent") is True
        ):
            product = self._resolve_order_product(message, criteria, context)
            if product is None:
                return self._ask_order_product(
                    conversation_id, context, criteria, message
                )
            return self._start_or_update_order_product(
                conversation_id,
                context,
                criteria,
                message,
                product,
            )

        if draft is None:
            return None

        if (
            pending_action in {"choose_order_size", "choose_order_color_and_size"}
            and criteria.get("height_cm") is not None
            and criteria.get("weight_kg") is not None
        ):
            return self._handle_order_size_recommendation(
                conversation_id, context, criteria, message
            )

        if pending_action in {
            "choose_order_color",
            "choose_order_size",
            "choose_order_color_and_size",
        }:
            return self._handle_order_variant_input(
                conversation_id,
                context,
                criteria,
                message,
            )

        if pending_action == "choose_order_quantity":
            quantity = extract_quantity(message)
            return self._handle_order_quantity(
                conversation_id,
                context,
                criteria,
                message,
                quantity,
            )

        if pending_action in {
            "collect_customer_name",
            "collect_customer_phone",
            "collect_customer_address",
        }:
            return self._handle_customer_info(
                conversation_id,
                context,
                criteria,
                message,
                pending_action,
            )

        if pending_action == "choose_payment_method":
            return self._handle_payment_method(
                conversation_id, context, criteria, message
            )

        if (
            criteria.get("intent") == "purchase_intent"
            or criteria.get("purchase_intent") is True
        ):
            product = self._resolve_order_product(message, criteria, context)
            if product is not None:
                return self._start_or_update_order_product(
                    conversation_id,
                    context,
                    criteria,
                    message,
                    product,
                )
            return self._continue_order_collection(conversation_id, context, criteria)

        return None

    def _save_order_context(
        self,
        conversation_id: str,
        context: dict[str, Any],
        pending_action: str | None,
    ) -> None:
        draft = context.get("order_draft")
        if isinstance(draft, dict):
            self._prepare_order_draft_metadata(draft, conversation_id)
        context["pending_action"] = pending_action
        CONVERSATION_STORE[conversation_id] = context

    def _prepare_order_draft_metadata(
        self,
        draft: dict[str, Any],
        conversation_id: str,
    ) -> None:

        if not draft.get("shop_id"):
            draft["shop_id"] = self.active_shop_id
        draft.setdefault("draft_order_id", draft.get("draft_id"))
        draft.setdefault("draft_id", draft.get("draft_order_id"))
        if not draft.get("customer_chat_id"):
            draft["customer_chat_id"] = conversation_id
        draft.setdefault("submitted_to_n8n", False)
        draft.setdefault("submission_in_progress", False)
        draft.setdefault("submitted_at", None)

    def _order_response(
        self,
        conversation_id: str,
        context: dict[str, Any],
        pending_action: str | None,
        criteria: dict[str, Any],
        reply: str,
        products: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        criteria = dict(criteria)
        criteria.setdefault("rag_contexts", [])
        criteria.setdefault("rag_status", self.rag_service.status())
        draft = context.get("order_draft")
        if draft:
            criteria["order_draft"] = draft
        self._save_order_context(conversation_id, context, pending_action)
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=products or [],
            rag_contexts=[],
        )

    def _resolve_order_product(
        self,
        message: str,
        criteria: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any] | None:
        product = self.search_engine.resolve_product_from_text(message, criteria)
        if product:
            return product

        last_product_ids = [
            str(product_id)
            for product_id in context.get("last_product_ids") or []
            if product_id
        ]
        index = product_reference_index(message, len(last_product_ids))
        if index is not None and index < len(last_product_ids):
            return self.search_engine.product_by_id(last_product_ids[index])

        selected_product_id = context.get("selected_product_id")
        if selected_product_id and (
            requested_current_product(message)
            or criteria.get("intent") == "purchase_intent"
        ):
            return self.search_engine.product_by_id(str(selected_product_id))

        if len(last_product_ids) == 1 and requested_current_product(message):
            return self.search_engine.product_by_id(last_product_ids[0])
        return None

    def _ask_order_product(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        last_error = getattr(self.search_engine, "last_error", None)
        if last_error:
            return self._order_response(
                conversation_id,
                context,
                context.get("pending_action"),
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "product_data_error",
                    "product_data_error": last_error,
                },
                SAFE_PRODUCT_DATA_ERROR_REPLY,
            )

        last_product_ids = context.get("last_product_ids") or []
        if (
            last_product_ids
            and len(last_product_ids) > 1
            and requested_current_product(message)
        ):
            reply = "Dạ, bạn muốn lấy mẫu nào trong các sản phẩm mình vừa tư vấn ạ?"
        else:
            reply = (
                "Dạ, bạn muốn mua mẫu sản phẩm nào ạ? Bạn có thể gửi tên mẫu hoặc chọn "
                "trong các sản phẩm mình vừa tư vấn."
            )
        return self._order_response(
            conversation_id,
            context,
            "choose_order_product",
            {
                **criteria,
                "intent": "purchase_intent",
                "response_mode": "ask_order_product",
            },
            reply,
        )

    def _is_freesize(self, size: str | None) -> bool:
        key = strip_accents(normalize_text(str(size or ""))).lower()
        return key.replace(" ", "") in {"freesize", "free", "fs", "sizefree", "sizefreesize"}

    def _check_order_variant_availability(
        self,
        *,
        product_id: Any,
        color: str | None,
        size: str | None,
    ) -> dict[str, Any]:
        criteria = {
            "selected_product_id": product_id,
            "color": color,
            "size": size,
            "shop_id": self.active_shop_id,
        }
        availability = self.search_engine.check_variant_availability(criteria)
        if availability.get("status") == "in_stock" or not self._is_freesize(size):
            return availability

        seen = {strip_accents(normalize_text(str(size or ""))).lower()}
        for alias in ["Freesize", "Free size", "FREE SIZE", "free size", "free", "FS"]:
            alias_key = strip_accents(normalize_text(alias)).lower()
            if alias_key in seen:
                continue
            seen.add(alias_key)
            retry = self.search_engine.check_variant_availability({**criteria, "size": alias})
            if retry.get("status") == "in_stock":
                return retry
            if retry.get("status") not in {"variant_not_found", "product_not_found"}:
                return retry
        return availability

    def _start_or_update_order_product(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
        product: dict[str, Any],
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        draft["status"] = "collecting"
        item = first_item(draft)
        if item.get("product_id") != product.get("product_id"):
            set_item_product(item, product)
            context["order_quantity_confirmed"] = False
        quantity = extract_quantity(message)
        if quantity is not None:
            item["quantity"] = quantity
            context["order_quantity_confirmed"] = True
        color, size = extract_order_color_size(message)
        if not color and context.get("color"):
            color = context.get("color")
        if not size and context.get("size"):
            size = context.get("size")
        if color:
            item["color"] = color
        if size:
            item["size"] = size
        context["selected_product_id"] = str(product.get("product_id"))
        return self._handle_order_variant_input(
            conversation_id,
            context,
            {
                **criteria,
                "intent": "purchase_intent",
                "response_mode": "purchase_intent",
                "selected_product_id": product.get("product_id"),
                "requested_product_name": product.get("product_name"),
            },
            message,
            already_extracted=True,
        )

    def _handle_order_variant_input(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
        *,
        already_extracted: bool = False,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        item = first_item(draft)
        if not item.get("product_id"):
            return self._ask_order_product(conversation_id, context, criteria, message)

        if not already_extracted:
            color, size = extract_order_color_size(message)
            if color:
                item["color"] = color
            if size:
                item["size"] = size

        if not item.get("color") and not item.get("size"):
            return self._order_response(
                conversation_id,
                context,
                "choose_order_color_and_size",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "ask_order_color_and_size",
                },
                "Dạ, bạn muốn lấy màu nào và size nào ạ?",
            )
        if not item.get("color"):
            return self._order_response(
                conversation_id,
                context,
                "choose_order_color",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "ask_order_color",
                },
                "Dạ, bạn muốn lấy màu nào ạ?",
            )
        if not item.get("size"):
            return self._order_response(
                conversation_id,
                context,
                "choose_order_size",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "ask_order_size",
                },
                "Dạ, bạn muốn lấy size nào ạ?",
            )

        availability = self._check_order_variant_availability(
            product_id=item.get("product_id"),
            color=item.get("color"),
            size=item.get("size"),
        )
        product = availability.get("product") or {}
        variant = availability.get("variant") or {}
        status = availability.get("status")
        criteria = {
            **criteria,
            "intent": "order_draft",
            "response_mode": f"order_variant_{status or 'not_found'}",
            "availability": availability,
        }

        if status != "in_stock":
            requested_color = item.get("color")
            requested_size = item.get("size")
            clear_item_variant(item, color=requested_color, size=None)
            reply = self._order_variant_unavailable_reply(
                product or {"product_name": item.get("product_name")},
                requested_color,
                requested_size,
                status,
                availability.get("alternatives") or [],
            )
            next_action = (
                "choose_order_size"
                if requested_color
                else "choose_order_color_and_size"
            )
            return self._order_response(
                conversation_id,
                context,
                next_action,
                criteria,
                reply,
            )

        set_item_variant(item, variant, product)
        item["unit_price"] = product.get("effective_price_vnd") or item.get(
            "unit_price"
        )
        recalculate_totals(draft)
        quantity = item.get("quantity")
        if context.get("order_quantity_confirmed") and quantity is not None:
            return self._handle_order_quantity(
                conversation_id,
                context,
                criteria,
                message,
                int(quantity),
            )
        return self._ask_order_quantity(conversation_id, context, criteria)

    def _ask_order_quantity(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        item = first_item(draft)
        reply = (
            f"Dạ, mẫu {item.get('product_name')} màu {str(item.get('color')).lower()} "
            f"size {item.get('size')} hiện còn {item.get('variant_stock')} sản phẩm. "
            "Bạn muốn lấy bao nhiêu cái ạ?"
        )
        return self._order_response(
            conversation_id,
            context,
            "choose_order_quantity",
            {
                **criteria,
                "intent": "order_draft",
                "response_mode": "order_variant_in_stock",
            },
            reply,
        )

    def _handle_order_quantity(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
        quantity: int | None,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        item = first_item(draft)
        stock = int(item.get("variant_stock") or 0)
        if quantity is None:
            return self._ask_order_quantity(conversation_id, context, criteria)
        if quantity < 1:
            context["order_quantity_confirmed"] = False
            return self._order_response(
                conversation_id,
                context,
                "choose_order_quantity",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "invalid_order_quantity",
                },
                "Dạ, số lượng cần lớn hơn 0. Bạn muốn lấy bao nhiêu sản phẩm ạ?",
            )
        if stock and quantity > stock:
            context["order_quantity_confirmed"] = False
            return self._order_response(
                conversation_id,
                context,
                "choose_order_quantity",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "order_quantity_exceeds_stock",
                },
                (
                    f"Dạ, mẫu này hiện chỉ còn {stock} sản phẩm. "
                    f"Bạn muốn lấy 1 hay {stock} sản phẩm ạ?"
                ),
            )
        item["quantity"] = quantity
        context["order_quantity_confirmed"] = True
        recalculate_totals(draft)
        return self._continue_order_collection(conversation_id, context, criteria)

    def _continue_order_collection(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        action = missing_order_action(draft)
        if action == "choose_order_quantity" and context.get(
            "order_quantity_confirmed"
        ):
            action = None
        if action:
            reply_by_action = {
                "choose_order_product": "Dạ, bạn muốn mua mẫu sản phẩm nào ạ?",
                "choose_order_color_and_size": "Dạ, bạn muốn lấy màu nào và size nào ạ?",
                "choose_order_color": "Dạ, bạn muốn lấy màu nào ạ?",
                "choose_order_size": "Dạ, bạn muốn lấy size nào ạ?",
                "choose_order_quantity": "Bạn muốn lấy bao nhiêu cái ạ?",
                "collect_customer_name": "Bạn cho mình xin tên người nhận ạ?",
                "collect_customer_phone": "Bạn cho mình xin số điện thoại nhận hàng ạ?",
                "collect_customer_address": "Bạn cho mình xin địa chỉ giao hàng đầy đủ ạ?",
                "choose_payment_method": "Hiện shop hỗ trợ thanh toán COD. Bạn muốn chọn COD chứ ạ?",
            }
            return self._order_response(
                conversation_id,
                context,
                action,
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": action.replace("choose_", "ask_"),
                },
                reply_by_action[action],
            )

        draft["status"] = "awaiting_customer_confirmation"
        recalculate_totals(draft)
        return self._order_response(
            conversation_id,
            context,
            "confirm_order_draft",
            {
                **criteria,
                "intent": "order_draft",
                "response_mode": "show_order_draft_summary",
            },
            build_summary(draft),
        )

    def _handle_customer_info(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
        pending_action: str,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        customer = draft.setdefault("customer", {})
        bundle = extract_customer_bundle(message)

        if pending_action == "collect_customer_name":
            if bundle.get("name"):
                customer["name"] = bundle["name"]
            if bundle.get("phone"):
                if not is_valid_phone(bundle["phone"]):
                    return self._order_response(
                        conversation_id,
                        context,
                        "collect_customer_phone",
                        {
                            **criteria,
                            "intent": "order_draft",
                            "response_mode": "invalid_customer_phone",
                        },
                        "Số điện thoại này chưa đúng định dạng. Bạn kiểm tra và gửi lại giúp mình nhé.",
                    )
                customer["phone"] = bundle["phone"]
            if bundle.get("address"):
                customer["address"] = bundle["address"]
            if not customer.get("name"):
                customer["name"] = message.strip()

        elif pending_action == "collect_customer_phone":
            phone = (
                bundle.get("phone")
                or extract_phone_candidate(message)
                or normalize_phone(message)
            )
            if not is_valid_phone(phone):
                return self._order_response(
                    conversation_id,
                    context,
                    "collect_customer_phone",
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "invalid_customer_phone",
                    },
                    "Số điện thoại này chưa đúng định dạng. Bạn kiểm tra và gửi lại giúp mình nhé.",
                )
            customer["phone"] = phone

        elif pending_action == "collect_customer_address":
            if bundle.get("address"):
                customer["address"] = bundle["address"]
            else:
                customer["address"] = message.strip()

        return self._continue_order_collection(conversation_id, context, criteria)

    def _handle_payment_method(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        method = normalize_payment_method(message)
        if method is None and text_key(message) in {
            "co",
            "ok",
            "oke",
            "duoc",
            "duoc a",
        }:
            method = "cod"
        if method == "bank_transfer":
            return self._order_response(
                conversation_id,
                context,
                "choose_payment_method",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "ask_payment_method",
                },
                "Hiện shop hỗ trợ thanh toán COD. Bạn muốn chọn COD chứ ạ?",
            )
        if method != "cod":
            return self._order_response(
                conversation_id,
                context,
                "choose_payment_method",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "ask_payment_method",
                },
                "Hiện shop hỗ trợ thanh toán COD. Bạn muốn chọn COD chứ ạ?",
            )
        draft["payment_method"] = "cod"
        return self._continue_order_collection(conversation_id, context, criteria)

    def _handle_order_edit(
        self,
        message: str,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
    ) -> dict[str, Any] | None:
        draft = active_order_draft(context)
        if not draft:
            return None
        key = text_key(message)
        item = first_item(draft)
        customer = draft.setdefault("customer", {})

        product = None
        if re.search(r"\bdoi\s+(?:sang\s+)?(?:mau|san pham|cai)\b", key):
            product = self._resolve_order_product(message, criteria, context)
            if product and product.get("product_id") != item.get("product_id"):
                set_item_product(item, product)
                context["order_quantity_confirmed"] = False
                return self._continue_order_collection(
                    conversation_id, context, criteria
                )

        if "doi mau" in key or "doi size" in key or "doi sang mau" in key:
            color, size = extract_order_color_size(message)
            requested_color = color or item.get("color")
            requested_size = size or item.get("size")
            if not requested_color or not requested_size:
                if color:
                    item["color"] = color
                if size:
                    item["size"] = size
                return self._continue_order_collection(
                    conversation_id, context, criteria
                )

            availability = self._check_order_variant_availability(
                product_id=item.get("product_id"),
                color=requested_color,
                size=requested_size,
            )
            if availability.get("status") != "in_stock":
                reply = self._order_variant_unavailable_reply(
                    availability.get("product")
                    or {"product_name": item.get("product_name")},
                    requested_color,
                    requested_size,
                    availability.get("status"),
                    availability.get("alternatives") or [],
                )
                return self._order_response(
                    conversation_id,
                    context,
                    "confirm_order_draft",
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "order_variant_not_available_on_edit",
                        "availability": availability,
                    },
                    reply,
                )
            set_item_variant(item, availability["variant"], availability.get("product"))
            item["unit_price"] = (availability.get("product") or {}).get(
                "effective_price_vnd"
            ) or item.get("unit_price")
            if int(item.get("quantity") or 1) > int(item.get("variant_stock") or 0):
                context["order_quantity_confirmed"] = False
                return self._order_response(
                    conversation_id,
                    context,
                    "choose_order_quantity",
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "order_quantity_exceeds_stock",
                    },
                    (
                        f"Dạ, mẫu này hiện chỉ còn {item.get('variant_stock')} sản phẩm. "
                        "Bạn muốn lấy số lượng bao nhiêu ạ?"
                    ),
                )
            recalculate_totals(draft)
            draft["status"] = "awaiting_customer_confirmation"
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "order_draft_updated",
                },
                build_summary(draft),
            )

        if "doi so luong" in key:
            quantity = extract_quantity(message)
            return self._handle_order_quantity(
                conversation_id,
                context,
                criteria,
                message,
                quantity,
            )

        if "doi so dien thoai" in key or "doi sdt" in key:
            phone = extract_phone_candidate(message) or normalize_phone(message)
            if not is_valid_phone(phone):
                return self._order_response(
                    conversation_id,
                    context,
                    "confirm_order_draft",
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "invalid_customer_phone",
                    },
                    "Số điện thoại này chưa đúng định dạng. Bạn kiểm tra và gửi lại giúp mình nhé.",
                )
            customer["phone"] = phone
        elif "doi dia chi" in key:
            value = re.sub(
                r"(?is)^.*?(?:đổi địa chỉ|doi dia chi)\s*(?:sang|thành|thanh)?\s*",
                "",
                message,
            ).strip()
            customer["address"] = value or message.strip()
        elif "doi ten" in key or "doi nguoi nhan" in key:
            value = re.sub(
                r"(?is)^.*?(?:đổi tên người nhận|đổi tên|đổi người nhận|doi ten nguoi nhan|doi ten|doi nguoi nhan)\s*(?:sang|thành|thanh)?\s*",
                "",
                message,
            ).strip()
            customer["name"] = value or message.strip()
        elif "doi thanh toan" in key:
            method = normalize_payment_method(message)
            if method == "bank_transfer":
                return self._order_response(
                    conversation_id,
                    context,
                    "confirm_order_draft",
                    {
                        **criteria,
                        "intent": "order_draft",
                        "response_mode": "ask_payment_method",
                    },
                    "Hiện shop hỗ trợ thanh toán COD. Bạn muốn chọn COD chứ ạ?",
                )
            if method == "cod":
                draft["payment_method"] = "cod"
            else:
                return None
        else:
            return None

        recalculate_totals(draft)
        draft["status"] = "awaiting_customer_confirmation"
        return self._order_response(
            conversation_id,
            context,
            "confirm_order_draft",
            {
                **criteria,
                "intent": "order_draft",
                "response_mode": "order_draft_updated",
            },
            build_summary(draft),
        )

    def _submit_order_draft(
        self,
        conversation_id: str,
        context: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        criteria = {
            "raw_message": message,
            "normalized_message": normalize_text(message),
            "intent": "order_draft",
            "response_mode": "confirm_order_draft",
            "need_clarification": False,
            "clarification_questions": [],
        }

        draft = active_order_draft(context)
        can_retry = is_retry(message) and bool(
            draft and draft.get("last_submission_failed")
        )
        if draft is None:
            return self._order_response(
                conversation_id,
                context,
                None,
                {
                    **criteria,
                    "response_mode": "order_draft_missing_for_confirmation",
                },
                "Mình chưa có đơn nháp nào đang chờ xác nhận. Bạn cho mình biết sản phẩm muốn mua trước nhé.",
            )

        self._prepare_order_draft_metadata(draft, conversation_id)

        if str(draft.get("customer_chat_id") or "") != str(conversation_id):
            return self._order_response(
                conversation_id,
                context,
                None,
                {
                    **criteria,
                    "response_mode": "order_draft_wrong_chat",
                },
                "Mình chưa có đơn nháp nào đang chờ xác nhận. Bạn cho mình biết sản phẩm muốn mua trước nhé.",
            )

        if (
            draft.get("submitted_to_n8n")
            or draft.get("status") == ORDER_STATE_PENDING_SHOP_APPROVAL
        ):
            draft["submission_in_progress"] = False
            return self._order_response(
                conversation_id,
                context,
                None,
                {
                    **criteria,
                    "response_mode": "order_already_submitted",
                },
                "Đơn của bạn đã được gửi sang shop và đang chờ duyệt rồi nhé.",
            )

        if (
            draft.get("submission_in_progress")
            or draft.get("status") == ORDER_STATE_SUBMITTING
        ):
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "response_mode": "order_submission_in_progress",
                },
                "Mình đang gửi đơn sang shop, bạn chờ mình một chút nhé.",
            )

        valid_confirmation = is_order_confirmation_message(message)
        if not valid_confirmation and not can_retry:
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "response_mode": "confirm_order_draft",
                },
                "Mình chưa có đơn nháp nào đang chờ xác nhận. Bạn cho mình biết sản phẩm muốn mua trước nhé.",
            )

        allowed_statuses = {ORDER_STATE_AWAITING_CONFIRMATION}
        if can_retry:
            allowed_statuses.add(ORDER_STATE_SUBMISSION_FAILED)
        if draft.get("status") not in allowed_statuses:
            return self._order_response(
                conversation_id,
                context,
                (
                    "confirm_order_draft"
                    if draft.get("status") == ORDER_STATE_SUBMISSION_FAILED
                    else None
                ),
                {
                    **criteria,
                    "response_mode": "order_draft_not_awaiting_confirmation",
                },
                "Mình chưa có đơn nháp nào đang chờ xác nhận. Bạn cho mình biết sản phẩm muốn mua trước nhé.",
            )

        recalculate_totals(draft)
        validation = validate_draft_order(draft)
        if not validation.ok:
            draft["status"] = ORDER_STATE_COLLECTING
            missing_text = ", ".join([*validation.missing_fields, *validation.errors])
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "response_mode": "order_draft_missing_required_fields",
                    "missing_fields": validation.missing_fields,
                    "validation_errors": validation.errors,
                },
                f"Mình chưa thể gửi đơn vì còn thiếu: {missing_text}. Bạn bổ sung giúp mình nhé.",
            )

        item = first_item(draft)
        availability = self._check_order_variant_availability(
            product_id=item.get("product_id"),
            color=item.get("color"),
            size=item.get("size"),
        )
        if availability.get("status") != "in_stock":
            draft["status"] = ORDER_STATE_AWAITING_CONFIRMATION
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "response_mode": "order_variant_changed_before_submit",
                    "availability": availability,
                },
                self._order_variant_unavailable_reply(
                    availability.get("product")
                    or {"product_name": item.get("product_name")},
                    item.get("color"),
                    item.get("size"),
                    availability.get("status"),
                    availability.get("alternatives") or [],
                ),
            )

        product = availability.get("product") or {}
        variant = availability.get("variant") or {}
        stock = int(variant.get("stock") or 0)
        if int(item.get("quantity") or 1) > stock:
            item["variant_stock"] = stock
            context["order_quantity_confirmed"] = False
            return self._order_response(
                conversation_id,
                context,
                "choose_order_quantity",
                {
                    **criteria,
                    "response_mode": "order_quantity_exceeds_stock",
                },
                (
                    f"Dạ, mẫu này hiện chỉ còn {stock} sản phẩm. "
                    f"Bạn muốn lấy 1 hay {stock} sản phẩm ạ?"
                ),
            )

        latest_price = product.get("effective_price_vnd")
        price_changed = latest_price is not None and int(latest_price) != int(
            item.get("unit_price") or 0
        )
        set_item_variant(item, variant, product)
        item["unit_price"] = latest_price
        recalculate_totals(draft)
        if price_changed:
            draft["status"] = ORDER_STATE_AWAITING_CONFIRMATION
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "response_mode": "show_order_draft_summary",
                },
                "Dạ, giá sản phẩm vừa được cập nhật. Bạn kiểm tra lại giúp mình nhé.\n\n"
                + build_summary(draft),
            )

        validation = validate_draft_order(draft)
        if not validation.ok:
            draft["status"] = ORDER_STATE_COLLECTING
            missing_text = ", ".join([*validation.missing_fields, *validation.errors])
            return self._order_response(
                conversation_id,
                context,
                "confirm_order_draft",
                {
                    **criteria,
                    "response_mode": "order_draft_missing_required_fields",
                    "missing_fields": validation.missing_fields,
                    "validation_errors": validation.errors,
                },
                f"Mình chưa thể gửi đơn vì còn thiếu: {missing_text}. Bạn bổ sung giúp mình nhé.",
            )

        draft["status"] = ORDER_STATE_SUBMITTING
        draft["submission_in_progress"] = True
        self._save_order_context(conversation_id, context, "confirm_order_draft")

        result = submit_draft_order_to_n8n(draft)
        if result.ok:
            submitted_at = datetime.now(timezone.utc).isoformat()
            draft["status"] = ORDER_STATE_PENDING_SHOP_APPROVAL
            draft["submitted_to_n8n"] = True
            draft["submission_in_progress"] = False
            draft["submitted_at"] = submitted_at
            draft["last_submission_failed"] = False
            context["last_submitted_order_payload"] = result.payload
            return self._order_response(
                conversation_id,
                context,
                None,
                {
                    **criteria,
                    "response_mode": "order_draft_submitted",
                    "webhook_status_code": result.status_code,
                },
                "Mình đã gửi đơn nháp sang shop để duyệt. Shop sẽ kiểm tra tồn kho và phản hồi lại bạn sớm nhé.",
            )

        draft["status"] = ORDER_STATE_SUBMISSION_FAILED
        draft["submitted_to_n8n"] = False
        draft["submission_in_progress"] = False
        draft["last_submission_failed"] = True
        return self._order_response(
            conversation_id,
            context,
            "confirm_order_draft",
            {
                **criteria,
                "response_mode": "order_draft_submission_failed",
                "webhook_error_type": result.error_type,
            },
            "Mình chưa thể gửi đơn sang shop lúc này. Thông tin đơn vẫn được giữ, bạn có thể nhắn “Gửi lại” để thử lại.",
        )

    def _cancel_order_draft(
        self,
        conversation_id: str,
        context: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context) if context.get("order_draft") else None
        if draft:
            draft["status"] = "cancelled"
        context["order_quantity_confirmed"] = False
        return self._order_response(
            conversation_id,
            context,
            None,
            {
                "raw_message": message,
                "normalized_message": normalize_text(message),
                "intent": "order_draft",
                "response_mode": "order_draft_cancelled",
                "need_clarification": False,
                "clarification_questions": [],
            },
            "Dạ, mình đã hủy đơn nháp này. Bạn có thể tiếp tục xem sản phẩm khác bất cứ lúc nào ạ.",
        )

    def _answer_policy_during_order(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        policy_key = str(criteria.get("policy_key") or "")
        policy_value = None
        if policy_key and not self.data_store.policies.empty:
            rows = self.data_store.policies[
                self.data_store.policies["policy_key"]
                .fillna("")
                .astype(str)
                .str.lower()
                == policy_key.lower()
            ]
            if not rows.empty:
                policy_value = str(rows.iloc[0].get("policy_value") or "").strip()
        if policy_value:
            reply = f"Dạ, theo chính sách hiện có của shop: {policy_value}."
        else:
            reply = "Dạ hiện dữ liệu chính sách của shop chưa có thông tin này, nên mình chưa thể khẳng định ạ."
        prompt = self._current_order_prompt(context)
        if prompt:
            reply = f"{reply}\n\nMình quay lại bước đang làm nhé: {prompt}"
        return self._order_response(
            conversation_id,
            context,
            context.get("pending_action"),
            {
                **criteria,
                "intent": "policy_question",
                "response_mode": "policy_question_during_order",
            },
            reply,
        )

    def _looks_like_order_stock_question(
        self, message: str, context: dict[str, Any]
    ) -> bool:
        if not active_order_draft(context):
            return False
        key = text_key(message)
        return bool(
            re.search(r"\b(?:con bao nhieu|con may|ton kho|con hang|con khong)\b", key)
        )

    def _answer_stock_during_order(
        self,
        conversation_id: str,
        context: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        item = first_item(draft)
        if item.get("sku"):
            availability = self._check_order_variant_availability(
                product_id=item.get("product_id"),
                color=item.get("color"),
                size=item.get("size"),
            )
            if availability.get("status") == "in_stock":
                set_item_variant(
                    item, availability["variant"], availability.get("product")
                )
                recalculate_totals(draft)
            stock = item.get("variant_stock")
            reply = (
                f"Dạ, mẫu {item.get('product_name')} màu {str(item.get('color')).lower()} "
                f"size {item.get('size')} hiện còn {stock} sản phẩm ạ."
            )
        else:
            reply = "Dạ, mình cần màu và size trước để kiểm tra đúng tồn kho variant cho bạn ạ."
        prompt = self._current_order_prompt(context)
        if prompt:
            reply = f"{reply}\n\n{prompt}"
        return self._order_response(
            conversation_id,
            context,
            context.get("pending_action"),
            {
                "raw_message": message,
                "normalized_message": normalize_text(message),
                "intent": "order_draft",
                "response_mode": "order_stock_question",
            },
            reply,
        )

    def _current_order_prompt(self, context: dict[str, Any]) -> str | None:
        action = context.get("pending_action")
        prompts = {
            "choose_order_product": "bạn muốn mua mẫu sản phẩm nào ạ?",
            "choose_order_color": "bạn muốn lấy màu nào ạ?",
            "choose_order_size": "bạn muốn lấy size nào ạ?",
            "choose_order_color_and_size": "bạn muốn lấy màu nào và size nào ạ?",
            "choose_order_quantity": "bạn muốn lấy bao nhiêu cái ạ?",
            "collect_customer_name": "bạn cho mình xin tên người nhận ạ?",
            "collect_customer_phone": "bạn cho mình xin số điện thoại nhận hàng ạ?",
            "collect_customer_address": "bạn cho mình xin địa chỉ giao hàng đầy đủ ạ?",
            "choose_payment_method": "hiện shop hỗ trợ COD, bạn muốn chọn COD chứ ạ?",
            "confirm_order_draft": "bạn kiểm tra đơn nháp, đúng thì nhắn “Xác nhận” giúp mình nhé.",
        }
        return prompts.get(str(action))

    def _order_variant_unavailable_reply(
        self,
        product: dict[str, Any],
        color: str | None,
        size: str | None,
        status: str | None,
        alternatives: list[dict[str, Any]],
    ) -> str:
        if status == "database_error":
            return SAFE_PRODUCT_DATA_ERROR_REPLY

        product_name = product.get("product_name") or "sản phẩm này"
        color_text = str(color or "").lower()
        size_text = str(size or "")
        if status == "out_of_stock":
            lines = [
                f"Dạ, mẫu {product_name} màu {color_text} size {size_text} hiện đang hết hàng ạ.",
                "",
            ]
            if alternatives:
                lines.append("Shop hiện còn:")
                for alternative in alternatives[:5]:
                    lines.append(
                        f"- Màu {str(alternative.get('color') or '').lower()} size {alternative.get('size')}"
                    )
                lines.extend(["", "Bạn muốn đổi sang lựa chọn nào ạ?"])
            return "\n".join(lines).strip()

        lines = [
            f"Dạ, mẫu này hiện chưa có màu {color_text} size {size_text} ạ.",
            "",
        ]
        if alternatives:
            lines.append("Các lựa chọn gần nhất đang có:")
            for alternative in alternatives[:5]:
                lines.append(
                    f"- {alternative.get('color')} size {alternative.get('size')}"
                )
        return "\n".join(lines).strip()

    def _browse_available_products_category(
        self,
        context: dict[str, Any],
        parsed: dict[str, Any],
    ) -> str | None:
        category = parsed.get("category_code") or context.get("category_code")
        if category:
            return str(category)

        dimensions = getattr(getattr(self.search_engine, "catalog_gate", None), "dimensions", None)
        summary = dimensions.summary() if dimensions and hasattr(dimensions, "summary") else {}
        categories = [str(item) for item in summary.get("category") or [] if item]
        return categories[0] if len(categories) == 1 else None

    def _available_products_browse_criteria(
        self,
        message: str,
        normalized_message: str,
        parsed: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        category_code = self._browse_available_products_category(context, parsed)
        criteria = {
            **parsed,
            "raw_message": message,
            "normalized_message": normalized_message,
            "intent": "product_search",
            "response_mode": "catalog_browsing",
            "category_code": category_code,
            "product_type": None,
            "product_types": [],
            "must_be_in_stock": True,
            "catalog_coverage": "supported",
            "need_clarification": False,
            "clarification_questions": [],
        }
        criteria["catalog_available_product_types"] = self._available_product_types(category_code)
        criteria.pop("out_of_stock_products", None)
        return criteria

    def _wants_available_products_browse(
        self,
        normalized_message: str,
        parsed: dict[str, Any],
    ) -> bool:
        if normalized_message in AFFIRMATIVE_FOLLOWUPS:
            return True
        if parsed.get("response_mode") == "catalog_browsing":
            return True
        if parsed.get("must_be_in_stock") is True:
            return True

        text = strip_accents(normalize_text(normalized_message))
        return any(
            phrase in text
            for phrase in [
                "con san pham nao",
                "con mau nao",
                "mau nao con",
                "san pham nao con",
                "xem san pham khac",
                "xem di",
                "goi y san pham khac",
                "goi y di",
            ]
        )

    def _handle_browse_available_products_followup(
        self,
        message: str,
        top_k: int,
        conversation_id: str,
        context: dict[str, Any],
        parsed: dict[str, Any],
        normalized_message: str,
    ) -> dict[str, Any] | None:
        if normalized_message in NEGATIVE_FOLLOWUPS:
            criteria = {
                "raw_message": message,
                "normalized_message": normalized_message,
                "intent": "conversation_followup",
                "response_mode": "browse_available_products_declined",
                "need_clarification": False,
                "clarification_questions": [],
            }
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply="Dạ vâng, mình không giới thiệu thêm sản phẩm khác lúc này ạ.",
                products=[],
                rag_contexts=[],
            )

        if not self._wants_available_products_browse(normalized_message, parsed):
            return None

        criteria = self._available_products_browse_criteria(
            message,
            normalized_message,
            parsed,
            context,
        )
        products = self.search_products(criteria, top_k=top_k)
        reply = build_reply(criteria, products)
        pending_action = self._store_pending_action(conversation_id, criteria, products)
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=products,
            rag_contexts=[],
        )

    def _handle_pending_followup(
        self, message: str, top_k: int, conversation_id: str
    ) -> dict[str, Any] | None:
        normalized_message = normalize_text(message)
        context = CONVERSATION_STORE.get(conversation_id)

        early_criteria = self.parse(message)
        if context and context.get("pending_action") == "confirm_cross_type_alternative":
            result = self._handle_cross_type_alternative_followup(
                message,
                top_k,
                conversation_id,
                context,
                early_criteria,
                normalized_message,
            )
            if result is not None:
                return result

        if early_criteria.get("intent") in {
            "policy_question",
            "negative_feedback",
            "reject_current_product",
            "request_alternative_product",
            "request_similar_product",
        }:
            return None

        if context and context.get("pending_action") == "browse_available_products":
            result = self._handle_browse_available_products_followup(
                message,
                top_k,
                conversation_id,
                context,
                early_criteria,
                normalized_message,
            )
            if result is not None:
                return result

        if context and context.get("pending_action") == "confirm_closest_product_type":
            if self._confirms_closest_product_type(normalized_message, context):
                criteria = self._criteria_from_closest_product_type_context(
                    message,
                    early_criteria,
                    context,
                )
                rag_contexts = self._attach_rag_contexts(message, criteria, [], top_k)
                products = self.search_products(criteria, top_k=top_k)
                products = self._rerank_products_with_rag(products, rag_contexts)
                if not products:
                    products = self._attach_nearest_over_budget(criteria)
                products = self._classify_empty_product_search(criteria, products)
                reply = build_reply(criteria, products)
                pending_action = self._store_pending_action(
                    conversation_id, criteria, products
                )
                return self._chat_payload(
                    conversation_id=conversation_id,
                    pending_action=pending_action,
                    criteria=criteria,
                    reply=reply,
                    products=products,
                    rag_contexts=rag_contexts,
                )

            if self._rejects_closest_product_type(normalized_message):
                criteria = {
                    "raw_message": message,
                    "normalized_message": normalized_message,
                    "intent": "conversation_followup",
                    "response_mode": "decline_closest_product_type",
                    "requested_product_group": context.get("requested_product_group"),
                    "suggested_product_type": context.get("suggested_product_type"),
                    "catalog_coverage": "closest_alternative_available",
                    "need_clarification": False,
                    "clarification_questions": [],
                    "rag_contexts": [],
                    "rag_status": self.rag_service.status(),
                }
                pending_action = self._store_pending_action(
                    conversation_id, criteria, []
                )
                return self._chat_payload(
                    conversation_id=conversation_id,
                    pending_action=pending_action,
                    criteria=criteria,
                    reply=build_reply(criteria, []),
                    products=[],
                    rag_contexts=[],
                )

        if context and context.get("pending_action") == "choose_fit_preference":
            parsed = early_criteria
            if parsed.get("fit_preference"):
                criteria = self._criteria_from_size_context(message, parsed, context)
                return self._handle_size_recommendation(
                    message, top_k, conversation_id, criteria
                )

        if context and context.get("pending_action") == "collect_size_profile":
            parsed = early_criteria
            if parsed.get("height_cm") or parsed.get("weight_kg"):
                criteria = self._criteria_from_size_context(message, parsed, context)
                return self._handle_size_recommendation(
                    message, top_k, conversation_id, criteria
                )

        if self._looks_like_size_request(normalized_message):
            criteria = self.parse(message)
            if criteria.get("intent") == "size_recommendation":
                return self._handle_size_recommendation(
                    message, top_k, conversation_id, criteria
                )

        if context and context.get("pending_action") == "choose_product_for_size":
            selected_product_type = self._canonical_fashion_product_type(
                normalized_message
            )
            if selected_product_type:
                criteria = self.parse(message)
                criteria.update(
                    {
                        "intent": "size_recommendation",
                        "category_code": context.get("category_code", "Fashion"),
                        "product_type": selected_product_type,
                        "product_types": [selected_product_type],
                        "height_cm": context.get("height_cm")
                        or criteria.get("height_cm"),
                        "weight_kg": context.get("weight_kg")
                        or criteria.get("weight_kg"),
                        "gender": criteria.get("gender") or context.get("gender"),
                        "fit_preference": criteria.get("fit_preference")
                        or context.get("fit_preference"),
                        "catalog_coverage": "supported",
                        "catalog_available_product_types": self._available_product_types(
                            "Fashion"
                        ),
                    }
                )
                return self._handle_size_recommendation(
                    message, top_k, conversation_id, criteria
                )

            criteria = {
                "raw_message": message,
                "normalized_message": normalized_message,
                "intent": "size_recommendation",
                "category_code": "Fashion",
                "catalog_available_product_types": self._available_product_types(
                    "Fashion"
                ),
                "response_mode": "ask_product_for_size",
                "height_cm": context.get("height_cm"),
                "weight_kg": context.get("weight_kg"),
                "gender": context.get("gender"),
                "fit_preference": context.get("fit_preference"),
                "need_clarification": True,
                "clarification_questions": [
                    "Bạn đang muốn chọn size cho áo thun, áo sơ mi hay sản phẩm nào ạ?"
                ],
            }
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        if context and context.get("pending_action") == "choose_product_for_variant":
            parsed = self.parse(message)
            criteria = self._merge_context_into_criteria(parsed, context)
            criteria["intent"] = "variant_availability_check"
            criteria = self._prepare_variant_criteria(message, criteria, context)
            return self._handle_variant_availability(
                message, top_k, conversation_id, criteria
            )

        if context and context.get("pending_action") == "choose_product_type":
            parsed = self.parse(message)
            if (
                parsed.get("response_mode") != "offer_closest_product_type"
                and parsed.get("product_type")
                and parsed.get("catalog_coverage") == "supported"
            ):
                fallback_probe = self.parse(f"tìm {message}")
                if fallback_probe.get("response_mode") == "offer_closest_product_type":
                    parsed = fallback_probe

            if parsed.get("response_mode") == "offer_closest_product_type":
                criteria = {
                    **parsed,
                    "raw_message": message,
                    "normalized_message": normalized_message,
                    "gender": parsed.get("gender") or context.get("gender"),
                    "use_case": parsed.get("use_case") or context.get("use_case"),
                    "style": parsed.get("style") or context.get("style"),
                    "color": parsed.get("color") or context.get("color"),
                    "budget_max_vnd": (
                        parsed.get("budget_max_vnd")
                        if parsed.get("budget_max_vnd") is not None
                        else context.get("budget_max_vnd")
                    ),
                    "fit_preference": (
                        parsed.get("fit_preference")
                        if parsed.get("fit_preference") is not None
                        else context.get("fit_preference")
                    ),
                    "product_type": None,
                    "product_types": [],
                    "need_clarification": True,
                    "clarification_questions": [],
                    "rag_contexts": [],
                    "rag_status": self.rag_service.status(),
                }
                pending_action = self._store_pending_action(
                    conversation_id, criteria, []
                )
                return self._chat_payload(
                    conversation_id=conversation_id,
                    pending_action=pending_action,
                    criteria=criteria,
                    reply=build_reply(criteria, []),
                    products=[],
                    rag_contexts=[],
                )

            if (
                parsed.get("product_type")
                and parsed.get("catalog_coverage") == "supported"
                and parsed.get("response_mode") != "offer_closest_product_type"
            ):
                selected_product_type = parsed.get("product_type")
                criteria = {
                    **parsed,
                    "raw_message": message,
                    "normalized_message": normalized_message,
                    "intent": "conversation_followup",
                    "category_code": parsed.get("category_code")
                    or context.get("category_code", "Fashion"),
                    "product_type": selected_product_type,
                    "product_types": [selected_product_type],
                    "catalog_coverage": "supported",
                    "catalog_available_product_types": self._available_product_types(),
                    "response_mode": "collect_product_preferences",
                    "missing_fields": PREFERENCE_FIELDS,
                    "need_clarification": True,
                    "clarification_questions": [
                        "Bạn muốn tìm cho nam/nữ/unisex, mặc đi làm hay đi chơi, và ngân sách khoảng bao nhiêu ạ?"
                    ],
                }
                criteria = self._merge_preference_context(
                    context, criteria, selected_product_type
                )
                pending_action = self._store_pending_action(
                    conversation_id, criteria, []
                )
                return self._chat_payload(
                    conversation_id=conversation_id,
                    pending_action=pending_action,
                    criteria=criteria,
                    reply=build_reply(criteria, []),
                    products=[],
                    rag_contexts=[],
                )

            criteria = {
                "raw_message": message,
                "normalized_message": normalized_message,
                "intent": "conversation_followup",
                "category_code": context.get("category_code", "Fashion"),
                "catalog_available_product_types": self._available_product_types(),
                "response_mode": "invalid_product_type_choice",
                "need_clarification": True,
                "clarification_questions": [
                    "Bạn muốn chọn nhóm nào trong các nhóm shop đang bán ạ?"
                ],
            }
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        if context and context.get("pending_action") == "choose_use_case":
            parsed = self.parse(message)
            if parsed.get("use_case"):
                criteria = {
                    "raw_message": message,
                    "normalized_message": normalized_message,
                    "intent": "conversation_followup",
                    "response_mode": "choose_product_type",
                    "category_code": "Fashion",
                    "use_case": parsed.get("use_case"),
                    "catalog_available_product_types": self._available_product_types(
                        "Fashion"
                    ),
                    "need_clarification": True,
                    "clarification_questions": [
                        "Bạn muốn mình tư vấn nhóm áo thun, áo sơ mi, váy, set bộ hay phụ kiện ạ?"
                    ],
                }
                pending_action = self._store_pending_action(
                    conversation_id, criteria, []
                )
                return self._chat_payload(
                    conversation_id=conversation_id,
                    pending_action=pending_action,
                    criteria=criteria,
                    reply=build_reply(criteria, []),
                    products=[],
                    rag_contexts=[],
                )

            criteria = {
                "raw_message": message,
                "normalized_message": normalized_message,
                "intent": "general_buying_intent",
                "response_mode": "ask_use_case",
                "category_code": "Fashion",
                "catalog_available_product_types": self._available_product_types(
                    "Fashion"
                ),
                "need_clarification": True,
                "clarification_questions": [
                    "Bạn đang cần đồ để đi làm, đi chơi, đi học hay mặc hằng ngày ạ?"
                ],
            }
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        if context and context.get("pending_action") == "collect_product_preferences":
            parsed = self.parse(message)
            selected_product_type = context.get("product_type")
            criteria = self._merge_preference_context(
                context, parsed, selected_product_type
            )
            if not self._has_collected_required_preferences(criteria):
                criteria.update(
                    {
                        "response_mode": "collect_product_preferences",
                        "intent": "conversation_followup",
                        "need_clarification": True,
                        "clarification_questions": self._preference_clarification_questions(
                            criteria
                        ),
                    }
                )
                pending_action = self._store_pending_action(
                    conversation_id, criteria, []
                )
                return self._chat_payload(
                    conversation_id=conversation_id,
                    pending_action=pending_action,
                    criteria=criteria,
                    reply=build_reply(criteria, []),
                    products=[],
                    rag_contexts=[],
                )

            criteria.update(
                {
                    "intent": "product_search",
                    "category_code": context.get("category_code", "Fashion"),
                    "product_type": selected_product_type,
                    "product_types": (
                        [selected_product_type] if selected_product_type else []
                    ),
                    "catalog_coverage": "supported",
                    "catalog_available_product_types": self._available_product_types(),
                    "response_mode": "product_search_with_collected_preferences",
                    "missing_fields": self._missing_collect_fields(criteria),
                    "need_clarification": False,
                    "clarification_questions": [],
                }
            )
            keywords = criteria.get("keywords") or []
            if selected_product_type and selected_product_type not in keywords:
                keywords.append(selected_product_type)
            criteria["keywords"] = keywords

            rag_contexts = self._attach_rag_contexts(message, criteria, [], top_k)
            products = self.search_products(criteria, top_k=top_k)
            products = self._rerank_products_with_rag(products, rag_contexts)
            if not products:
                products = self._attach_nearest_over_budget(criteria)
            products = self._classify_empty_product_search(criteria, products)
            reply = build_reply(criteria, products)
            pending_action = self._store_pending_action(
                conversation_id, criteria, products
            )
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=reply,
                products=products,
                rag_contexts=rag_contexts,
            )

        if normalized_message in AFFIRMATIVE_FOLLOWUPS:
            criteria = {
                "raw_message": message,
                "normalized_message": normalized_message,
                "intent": "conversation_followup",
                "response_mode": "choose_product_type",
                "category_code": "Fashion",
                "catalog_available_product_types": self._available_product_types(
                    "Fashion"
                ),
                "need_clarification": True,
                "clarification_questions": [
                    "Bạn muốn mình gợi ý sản phẩm thuộc nhóm nào ạ?"
                ],
            }
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        return None

    def _handle_cross_type_alternative_followup(
        self,
        message: str,
        top_k: int,
        conversation_id: str,
        context: dict[str, Any],
        parsed: dict[str, Any],
        normalized_message: str,
    ) -> dict[str, Any] | None:
        if self._rejects_cross_type_alternative(normalized_message):
            criteria = {
                **parsed,
                "raw_message": message,
                "normalized_message": normalized_message,
                "intent": "conversation_followup",
                "response_mode": "cross_type_alternative_declined",
                "need_clarification": False,
                "clarification_questions": [],
            }
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply="Dạ vâng, mình chưa mở rộng sang nhóm sản phẩm khác lúc này ạ.",
                products=[],
                rag_contexts=[],
            )

        if not self._confirms_cross_type_alternative(normalized_message):
            return None

        criteria = self._criteria_from_cross_type_alternative_context(
            message,
            parsed,
            context,
            normalized_message,
        )
        self._drop_duplicate_style_use_case(criteria)
        rag_contexts = self._attach_rag_contexts(message, criteria, [], top_k)
        products = self.search_products(criteria, top_k=top_k)
        products = self._rerank_products_with_rag(products, rag_contexts)
        if not products:
            products = self._attach_nearest_over_budget(criteria)
        products = self._classify_empty_product_search(criteria, products)
        reply = build_reply(criteria, products)
        pending_action = self._store_pending_action(conversation_id, criteria, products)
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=products,
            rag_contexts=rag_contexts,
        )

    def _confirms_cross_type_alternative(self, normalized_message: str) -> bool:
        text_key = strip_accents(normalized_message)
        if normalized_message in AFFIRMATIVE_FOLLOWUPS:
            return True
        if text_key in {"co", "ok", "oke", "duoc", "uh", "uhm", "um"}:
            return True
        return any(
            phrase in text_key
            for phrase in [
                "xem di",
                "goi y di",
                "cho xem",
                "san pham khac cung duoc",
            ]
        )

    def _rejects_cross_type_alternative(self, normalized_message: str) -> bool:
        text_key = strip_accents(normalized_message)
        if normalized_message in NEGATIVE_FOLLOWUPS:
            return True
        return any(
            phrase == text_key or phrase in text_key
            for phrase in ["khong", "thoi", "khoi", "chua"]
        )

    def _criteria_from_cross_type_alternative_context(
        self,
        message: str,
        parsed: dict[str, Any],
        context: dict[str, Any],
        normalized_message: str,
    ) -> dict[str, Any]:
        rejected_ids = self._cross_type_excluded_product_ids(context)
        criteria = {
            **parsed,
            "raw_message": message,
            "normalized_message": normalized_message,
            "intent": "product_search",
            "response_mode": "cross_type_alternative",
            "category_code": context.get("category_code") or "Fashion",
            "product_type": None,
            "product_types": [],
            "gender": parsed.get("gender") or context.get("gender"),
            "use_case": parsed.get("use_case") or context.get("use_case"),
            "budget_max_vnd": (
                parsed.get("budget_max_vnd")
                if parsed.get("budget_max_vnd") is not None
                else context.get("budget_max_vnd")
            ),
            "color": parsed.get("color") or context.get("color"),
            "size": parsed.get("size") or context.get("size"),
            "style": parsed.get("style") or context.get("style"),
            "exclude_product_ids": rejected_ids,
            "rejected_product_ids": rejected_ids,
            "selected_product_id": None,
            "selected_variant_sku": None,
            "catalog_coverage": "supported",
            "must_be_in_stock": True,
            "need_clarification": False,
            "clarification_questions": [],
        }
        return criteria

    def _cross_type_excluded_product_ids(self, context: dict[str, Any]) -> list[str]:
        product_ids = [
            str(product_id)
            for product_id in [
                *(context.get("rejected_product_ids") or []),
                context.get("selected_product_id"),
            ]
            if product_id
        ]
        return list(dict.fromkeys(product_ids))

    def _prepare_contextual_criteria(
        self,
        message: str,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        context = CONVERSATION_STORE.get(conversation_id) or {}
        criteria = dict(criteria)
        if self._should_reset_product_context(message, criteria):
            context = {}
            criteria["reset_product_context"] = True
            criteria["exclude_product_ids"] = []
        else:
            criteria["exclude_product_ids"] = self._excluded_product_ids(context)

        if self._is_price_only_followup(criteria, context):
            criteria = self._criteria_from_price_followup(criteria, context)

        if criteria.get("intent") in {
            "policy_question",
            "negative_feedback",
            "reject_current_product",
            "request_alternative_product",
            "request_similar_product",
            "variant_availability_check",
        }:
            criteria = self._merge_context_into_criteria(criteria, context)

        if criteria.get(
            "intent"
        ) == "variant_availability_check" or self._should_check_variant_availability(
            message, criteria, context
        ):
            return self._prepare_variant_criteria(message, criteria, context)

        product = (
            None
            if criteria.get("price_followup")
            else self.search_engine.resolve_product_from_text(message, criteria)
        )
        if product and self._looks_like_direct_product_availability(criteria):
            criteria.update(
                {
                    "intent": (
                        "mixed_product_request"
                        if criteria.get("out_of_scope_items")
                        else "product_availability_check"
                    ),
                    "response_mode": "variant_availability",
                    "requested_product_name": product.get("product_name"),
                    "selected_product_id": product.get("product_id"),
                    "product_type": product.get("product_type")
                    or criteria.get("product_type"),
                    "product_types": (
                        [product.get("product_type")]
                        if product.get("product_type")
                        else criteria.get("product_types", [])
                    ),
                    "category_code": product.get("category_code")
                    or criteria.get("category_code"),
                    "catalog_coverage": (
                        "partially_supported"
                        if criteria.get("out_of_scope_items")
                        else "supported"
                    ),
                    "need_clarification": False,
                    "clarification_questions": [],
                }
            )
        return criteria

    def _is_price_filter_direction(self, direction: Any) -> bool:
        return str(direction or "") in PRICE_FILTER_DIRECTIONS

    def _has_explicit_price_filter(self, criteria: dict[str, Any]) -> bool:
        return bool(
            criteria.get("budget_min_vnd") is not None
            or criteria.get("budget_max_vnd") is not None
            or self._is_price_filter_direction(criteria.get("price_direction"))
        )

    def _has_valid_requested_product_group(self, criteria: dict[str, Any]) -> bool:
        group = criteria.get("requested_product_group")
        if not group:
            return False
        group_key = strip_accents(normalize_text(str(group)))
        generic_keys = {
            strip_accents(normalize_text(value))
            for value in GENERIC_QUESTION_PRODUCT_GROUPS
        }
        return group_key not in generic_keys

    def _is_price_only_followup(
        self,
        criteria: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        if not self._has_explicit_price_filter(criteria):
            return False
        if criteria.get("product_type") or criteria.get("requested_product_name"):
            return False
        if self._has_valid_requested_product_group(criteria):
            return False
        return bool(
            context.get("product_type")
            or context.get("last_product_type")
            or context.get("selected_product_id")
            or context.get("last_product_ids")
        )

    def _criteria_from_price_followup(
        self,
        parsed: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        product_type = context.get("product_type") or context.get("last_product_type")
        criteria = {
            **parsed,
            "intent": "conversation_followup",
            "category_code": (
                parsed.get("category_code") or context.get("category_code") or "Fashion"
            ),
            "product_type": product_type,
            "product_types": [product_type] if product_type else [],
            "gender": parsed.get("gender") or context.get("gender"),
            "use_case": parsed.get("use_case") or context.get("use_case"),
            "style": parsed.get("style") or context.get("style"),
            "color": parsed.get("color") or context.get("color"),
            "size": parsed.get("size") or context.get("size"),
            "requested_product_group": None,
            "out_of_scope_items": [],
            "catalog_coverage": "supported",
            "price_followup": True,
            "need_clarification": False,
            "clarification_questions": [],
        }
        self._apply_price_context(criteria, context)
        return criteria

    def _apply_price_context(
        self,
        criteria: dict[str, Any],
        context: dict[str, Any],
    ) -> None:
        if not self._has_explicit_price_filter(criteria):
            for field in ["budget_min_vnd", "budget_max_vnd", "price_direction"]:
                if criteria.get(field) is None and context.get(field) is not None:
                    criteria[field] = context.get(field)
            return

        direction = criteria.get("price_direction")
        if direction == "between":
            return
        if criteria.get("budget_min_vnd") is not None:
            criteria["budget_max_vnd"] = None
            return
        if criteria.get("budget_max_vnd") is not None:
            criteria["budget_min_vnd"] = None

    def _should_reset_product_context(
        self, message: str, criteria: dict[str, Any]
    ) -> bool:
        text = strip_accents(normalize_text(message))
        if not any(
            phrase in text
            for phrase in [
                "doi san pham",
                "doi sang san pham",
                "san pham khac",
                "chuyen san pham",
            ]
        ):
            return False
        return bool(
            criteria.get("product_type")
            or criteria.get("requested_product_name")
            or criteria.get("color")
            or criteria.get("gender")
            or criteria.get("use_case")
        )

    def _merge_context_into_criteria(
        self,
        criteria: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(criteria)
        for field in [
            "category_code",
            "product_type",
            "gender",
            "use_case",
            "style",
            "color",
            "size",
            "height_cm",
            "weight_kg",
            "fit_preference",
        ]:
            if merged.get(field) is None and context.get(field) is not None:
                merged[field] = context.get(field)
        self._apply_price_context(merged, context)

        if merged.get("product_type") and not merged.get("product_types"):
            merged["product_types"] = [merged["product_type"]]
        if context.get("selected_product_id") and not merged.get("selected_product_id"):
            merged["selected_product_id"] = context.get("selected_product_id")
        return merged

    def _criteria_from_size_context(
        self,
        message: str,
        parsed: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        product_type = (
            parsed.get("product_type")
            or context.get("product_type")
            or context.get("last_product_type")
        )

        criteria = {
            **parsed,
            "raw_message": message,
            "intent": "size_recommendation",
            "response_mode": "size_recommendation",
            "category_code": (
                parsed.get("category_code") or context.get("category_code") or "Fashion"
            ),
            "product_type": product_type,
            "product_types": [product_type] if product_type else [],
            "gender": parsed.get("gender") or context.get("gender"),
            "use_case": parsed.get("use_case") or context.get("use_case"),
            "height_cm": (
                parsed.get("height_cm")
                if parsed.get("height_cm") is not None
                else context.get("height_cm")
            ),
            "weight_kg": (
                parsed.get("weight_kg")
                if parsed.get("weight_kg") is not None
                else context.get("weight_kg")
            ),
            "fit_preference": (
                parsed.get("fit_preference")
                if parsed.get("fit_preference") is not None
                else context.get("fit_preference")
            ),
            "need_clarification": False,
            "clarification_questions": [],
        }

        criteria.pop("size_recommendation", None)
        criteria.pop("recommended_size", None)
        criteria.pop("alternative_size", None)

        return criteria

    def _excluded_product_ids(self, context: dict[str, Any]) -> list[str]:
        return [
            str(product_id)
            for product_id in context.get("rejected_product_ids") or []
            if product_id
        ]

    def _looks_like_direct_product_availability(self, criteria: dict[str, Any]) -> bool:
        return bool(
            criteria.get("must_be_in_stock")
            or criteria.get("variant_followup")
            or (
                (criteria.get("size") or criteria.get("color"))
                and criteria.get("response_mode") == "variant_availability"
            )
        )

    def _should_check_variant_availability(
        self,
        message: str,
        criteria: dict[str, Any],
        context: dict[str, Any],
    ) -> bool:
        if criteria.get("intent") in {
            "product_availability_check",
            "mixed_product_request",
            "variant_availability_check",
        }:
            return True
        if criteria.get("intent") in {
            "reject_current_product",
            "request_alternative_product",
            "request_similar_product",
        }:
            return False
        if criteria.get("product_type") and (
            criteria.get("budget_min_vnd") is not None
            or criteria.get("budget_max_vnd") is not None
            or criteria.get("price_direction") in PRICE_FILTER_DIRECTIONS
            or criteria.get("gender")
            or criteria.get("use_case")
        ):
            return False
        if not context.get("selected_product_id") and not context.get(
            "last_product_ids"
        ):
            return False

        text_key = strip_accents(normalize_text(message))
        if "mau khac" in text_key or "size khac" in text_key:
            return True
        if re.search(r"\b(?:co|con)\s+(?:mau|size)\b", text_key):
            return True
        if criteria.get("must_be_in_stock") and (
            criteria.get("size") or criteria.get("color")
        ):
            return True
        return False

    def _prepare_variant_criteria(
        self,
        message: str,
        criteria: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        prepared = self._merge_context_into_criteria(criteria, context)
        product = self.search_engine.resolve_product_from_text(message, prepared)
        resolved_from_context = False
        product_id = prepared.get("selected_product_id") or context.get(
            "selected_product_id"
        )
        if not product_id and context.get("last_product_ids"):
            product_id = context["last_product_ids"][0]
        if product is None and product_id:
            product = self.search_engine.product_by_id(str(product_id))
            resolved_from_context = True

        requested_size = self._requested_variant_size(message)
        if requested_size:
            prepared["size"] = requested_size
        if "mau khac" in strip_accents(normalize_text(message)):
            prepared["color"] = None
            prepared["response_mode"] = "variant_options"

        if product is None:
            prepared.update(
                {
                    "intent": "variant_availability_check",
                    "response_mode": "ask_product_for_variant",
                    "need_clarification": True,
                    "clarification_questions": [
                        "Bạn muốn mình kiểm tra màu/size này cho mẫu sản phẩm nào ạ?"
                    ],
                    "rag_contexts": [],
                    "rag_status": self.rag_service.status(),
                }
            )
            return prepared

        if prepared.get("out_of_scope_items"):
            prepared_intent = "mixed_product_request"
        elif prepared.get("intent") == "product_availability_check" or (
            not resolved_from_context and prepared.get("must_be_in_stock")
        ):
            prepared_intent = "product_availability_check"
        else:
            prepared_intent = "variant_availability_check"

        prepared.update(
            {
                "intent": prepared_intent,
                "response_mode": prepared.get("response_mode")
                or "variant_availability",
                "requested_product_name": product.get("product_name"),
                "selected_product_id": product.get("product_id"),
                "product_type": product.get("product_type")
                or prepared.get("product_type"),
                "product_types": (
                    [product.get("product_type")]
                    if product.get("product_type")
                    else prepared.get("product_types", [])
                ),
                "category_code": product.get("category_code")
                or prepared.get("category_code"),
                "catalog_coverage": (
                    "partially_supported"
                    if prepared.get("out_of_scope_items")
                    else "supported"
                ),
                "need_clarification": False,
                "clarification_questions": [],
            }
        )
        return prepared

    def _requested_variant_size(self, message: str) -> str | None:
        text_key = strip_accents(normalize_text(message)).upper()
        patterns = [
            r"\bCO\s+SIZE\s*(XL|L|M|S)\b",
            r"\bCON\s+SIZE\s*(XL|L|M|S)\b",
            r"\bDOI\s+(?:SANG\s+)?SIZE\s*(XL|L|M|S)\b",
            r"\bLAY\s+SIZE\s*(XL|L|M|S)\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, text_key)
            if match:
                return match.group(1)
        return None

    def _handle_closest_product_type_offer(
        self,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        criteria = dict(criteria)
        criteria["rag_contexts"] = []
        criteria["rag_status"] = self.rag_service.status()
        pending_action = self._store_pending_action(conversation_id, criteria, [])
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=build_reply(criteria, []),
            products=[],
            rag_contexts=[],
        )

    def _criteria_from_closest_product_type_context(
        self,
        message: str,
        parsed: dict[str, Any],
        context: dict[str, Any],
    ) -> dict[str, Any]:
        suggested_product_type = context.get("suggested_product_type")
        criteria = {
            **parsed,
            "raw_message": message,
            "intent": "product_search",
            "response_mode": "product_search",
            "category_code": (
                parsed.get("category_code") or context.get("category_code") or "Fashion"
            ),
            "product_type": suggested_product_type,
            "product_types": [suggested_product_type] if suggested_product_type else [],
            "requested_product_group": context.get("requested_product_group"),
            "suggested_product_type": suggested_product_type,
            "catalog_coverage": "supported",
            "catalog_available_product_types": self._available_product_types(),
            "gender": parsed.get("gender") or context.get("gender"),
            "use_case": parsed.get("use_case") or context.get("use_case"),
            "style": parsed.get("style") or context.get("style"),
            "color": parsed.get("color") or context.get("color"),
            "budget_max_vnd": (
                parsed.get("budget_max_vnd")
                if parsed.get("budget_max_vnd") is not None
                else context.get("budget_max_vnd")
            ),
            "size": parsed.get("size") or context.get("size"),
            "fit_preference": (
                parsed.get("fit_preference")
                if parsed.get("fit_preference") is not None
                else context.get("fit_preference")
            ),
            "out_of_scope_items": [],
            "need_clarification": False,
            "clarification_questions": [],
        }
        criteria["catalog_gate"] = {
            "status": "supported",
            "category_code": criteria.get("category_code"),
            "product_types": criteria.get("product_types") or [],
            "unsupported_terms": [],
            "requested_product_concepts": [],
            "matched_chunks": [],
        }

        keywords = list(dict.fromkeys(criteria.get("keywords") or []))
        for value in [
            suggested_product_type,
            criteria.get("gender"),
            criteria.get("use_case"),
            criteria.get("style"),
            criteria.get("color"),
        ]:
            if value and value not in keywords:
                keywords.append(value)
        criteria["keywords"] = keywords
        return criteria

    def _confirms_closest_product_type(
        self,
        normalized_message: str,
        context: dict[str, Any],
    ) -> bool:
        if self._rejects_closest_product_type(normalized_message):
            return False

        message_key = strip_accents(normalize_text(normalized_message))
        if normalized_message in AFFIRMATIVE_FOLLOWUPS or message_key in {
            strip_accents(normalize_text(value)) for value in AFFIRMATIVE_FOLLOWUPS
        }:
            return True

        if re.search(r"\b(co|duoc|ok|oke|uh|uhm|xem|gioi thieu|goi y)\b", message_key):
            return True

        suggested_product_type = context.get("suggested_product_type")
        if suggested_product_type:
            suggested_key = strip_accents(normalize_text(str(suggested_product_type)))
            return bool(suggested_key and suggested_key in message_key)
        return False

    def _rejects_closest_product_type(self, normalized_message: str) -> bool:
        message_key = strip_accents(normalize_text(normalized_message))
        negative_keys = {
            strip_accents(normalize_text(value)) for value in NEGATIVE_FOLLOWUPS
        }
        if normalized_message in NEGATIVE_FOLLOWUPS or message_key in negative_keys:
            return True
        return bool(
            re.search(
                r"\b(khong|khong can|thoi|khoi|chua|de sau|ko)\b",
                message_key,
            )
        )

    def _handle_policy_question(
        self,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        criteria = dict(criteria)
        policy_key = str(criteria.get("policy_key") or "")
        policy_value = None
        if policy_key and not self.data_store.policies.empty:
            rows = self.data_store.policies[
                self.data_store.policies["policy_key"]
                .fillna("")
                .astype(str)
                .str.lower()
                == policy_key.lower()
            ]
            if not rows.empty:
                policy_value = str(rows.iloc[0].get("policy_value") or "").strip()

        criteria["policy"] = {
            "policy_key": policy_key,
            "policy_label": POLICY_LABELS.get(policy_key, policy_key),
            "policy_value": policy_value,
        }
        criteria["rag_contexts"] = []
        criteria["rag_status"] = self.rag_service.status()
        pending_action = self._store_pending_action(conversation_id, criteria, [])
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=build_reply(criteria, []),
            products=[],
            rag_contexts=[],
        )

    def _handle_negative_feedback(
        self,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        criteria = dict(criteria)
        criteria["rag_contexts"] = []
        criteria["rag_status"] = self.rag_service.status()
        pending_action = self._store_pending_action(conversation_id, criteria, [])
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=build_reply(criteria, []),
            products=[],
            rag_contexts=[],
        )

    def _handle_variant_availability(
        self,
        message: str,
        top_k: int,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        criteria = dict(criteria)
        if criteria.get("response_mode") == "ask_product_for_variant":
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        products = self._handle_structured_availability(criteria)
        rag_contexts = self._attach_rag_contexts(message, criteria, products, top_k)
        reply = build_reply(criteria, products)
        pending_action = self._store_pending_action(conversation_id, criteria, products)
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=products,
            rag_contexts=rag_contexts,
        )

    def _handle_alternative_product(
        self,
        message: str,
        top_k: int,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        context = CONVERSATION_STORE.get(conversation_id) or {}
        criteria = self._merge_context_into_criteria(dict(criteria), context)
        current_product = self._current_product(context)
        rejected_ids = self._excluded_product_ids(context)
        current_product_id = str(
            (current_product or {}).get("product_id")
            or context.get("selected_product_id")
            or ""
        )

        if (
            self._is_alternative_product_request(criteria)
            and current_product_id
            and current_product_id not in rejected_ids
        ):
            rejected_ids.append(current_product_id)

        exclude_ids = list(dict.fromkeys([*rejected_ids]))
        if current_product_id and criteria.get("response_mode") in {
            "cheaper_product",
            "premium_product",
            "style_change",
        }:
            exclude_ids = list(dict.fromkeys([*exclude_ids, current_product_id]))

        criteria["exclude_product_ids"] = exclude_ids
        criteria["rejected_product_ids"] = rejected_ids
        criteria["catalog_coverage"] = "supported"
        criteria["category_code"] = (
            criteria.get("category_code") or context.get("category_code") or "Fashion"
        )
        if criteria.get("product_type") and not criteria.get("product_types"):
            criteria["product_types"] = [criteria["product_type"]]

        if criteria.get("response_mode") == "cheaper_product" and current_product:
            current_price = int(current_product.get("effective_price_vnd") or 0)
            if current_price > 0:
                existing_budget = criteria.get("budget_max_vnd")
                criteria["budget_max_vnd"] = (
                    min(int(existing_budget), current_price - 1)
                    if existing_budget
                    else current_price - 1
                )
                criteria["current_price_vnd"] = current_price

        if criteria.get("response_mode") == "premium_product":
            criteria["current_price_vnd"] = int(
                (current_product or {}).get("effective_price_vnd") or 0
            )
            criteria["budget_max_vnd"] = None

        if criteria.get("response_mode") == "style_change":
            criteria["style"] = (
                criteria.get("style") or criteria.get("style_change") or "công sở"
            )
            keywords = list(criteria.get("keywords") or [])
            for keyword in [criteria["style"], criteria.get("use_case")]:
                if keyword and keyword not in keywords:
                    keywords.append(keyword)
            criteria["keywords"] = keywords

        self._drop_duplicate_style_use_case(criteria)
        rag_contexts = self._attach_rag_contexts(message, criteria, [], top_k)
        products = self.search_products(criteria, top_k=top_k)
        if criteria.get("response_mode") == "premium_product":
            products.sort(
                key=lambda product: (
                    float(product.get("rating") or 0),
                    int(product.get("sold_30d") or 0),
                    int(product.get("effective_price_vnd") or 0),
                ),
                reverse=True,
            )
        products = self._rerank_products_with_rag(products, rag_contexts)
        if (
            not products
            and self._is_alternative_product_request(criteria)
            and criteria.get("product_type")
            and not criteria.get("product_data_error")
        ):
            criteria["response_mode"] = "same_type_alternative_not_found"
            criteria["same_type_product_type"] = criteria["product_type"]
            criteria["need_clarification"] = True
            criteria["clarification_questions"] = [
                "Bạn có muốn xem sản phẩm thuộc nhóm khác nhưng vẫn phù hợp với nhu cầu hiện tại không?"
            ]
            reply = build_reply(criteria, products)
            pending_action = self._store_pending_action(
                conversation_id, criteria, products
            )
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=reply,
                products=products,
                rag_contexts=rag_contexts,
            )
        if not products:
            products = self._attach_nearest_over_budget(criteria)
        products = self._classify_empty_product_search(criteria, products)

        reply = build_reply(criteria, products)
        pending_action = self._store_pending_action(conversation_id, criteria, products)
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=products,
            rag_contexts=rag_contexts,
        )

    def _is_alternative_product_request(self, criteria: dict[str, Any]) -> bool:
        return bool(
            criteria.get("intent") in ALTERNATIVE_PRODUCT_INTENTS
            or criteria.get("response_mode") in ALTERNATIVE_PRODUCT_RESPONSE_MODES
        )

    def _drop_duplicate_style_use_case(self, criteria: dict[str, Any]) -> None:
        if (
            criteria.get("style")
            and criteria.get("use_case")
            and normalize_text(str(criteria["style"]))
            == normalize_text(str(criteria["use_case"]))
        ):
            criteria["style"] = None

    def _current_product(self, context: dict[str, Any]) -> dict[str, Any] | None:
        product_id = context.get("selected_product_id")
        if not product_id and context.get("last_product_ids"):
            product_id = context["last_product_ids"][0]
        if not product_id:
            return None
        return self.search_engine.product_by_id(str(product_id))

    def _should_start_preference_collection(self, criteria: dict[str, Any]) -> bool:
        if criteria.get("intent") != "product_search":
            return False
        if criteria.get("response_mode") in {"catalog_browsing", "catalog_overview"}:
            return False
        if criteria.get("requested_product_name"):
            return False
        if not criteria.get("product_type"):
            return False
        message_text = str(
            criteria.get("raw_message") or criteria.get("normalized_message") or ""
        )
        if self.search_engine.resolve_product_from_text(message_text, criteria):
            return False
        text_key = strip_accents(
            normalize_text(
                criteria.get("normalized_message") or criteria.get("raw_message")
            )
        )
        if "co " in f"{text_key} " and (
            " nao" in f" {text_key}" or " khong" in f" {text_key}"
        ):
            return False

        has_gender = bool(criteria.get("gender"))
        has_use_case = bool(criteria.get("use_case"))
        has_budget = (
            criteria.get("budget_min_vnd") is not None
            or criteria.get("budget_max_vnd") is not None
        )
        has_color_or_size = bool(criteria.get("color") or criteria.get("size"))
        has_style = bool(criteria.get("style"))

        if not any(
            [has_gender, has_use_case, has_budget, has_color_or_size, has_style]
        ):
            return True
        if has_gender and not has_use_case and not has_budget and not has_color_or_size:
            return True
        if has_use_case and not has_gender and not has_budget and not has_color_or_size:
            return True
        return False

    def _prepare_collect_preferences_criteria(
        self,
        criteria: dict[str, Any],
        conversation_id: str,
    ) -> dict[str, Any]:
        existing_context = CONVERSATION_STORE.get(conversation_id) or {}
        context = (
            existing_context
            if existing_context.get("pending_action")
            in {"collect_product_preferences", "choose_product_type"}
            else {}
        )
        product_type = (
            criteria.get("product_type")
            or context.get("product_type")
            or context.get("last_product_type")
        )
        merged = self._merge_preference_context(context, criteria, product_type)
        merged.update(
            {
                "intent": "conversation_followup",
                "response_mode": "collect_product_preferences",
                "need_clarification": True,
                "clarification_questions": self._preference_clarification_questions(
                    merged
                ),
            }
        )
        return merged

    def _merge_preference_context(
        self,
        context: dict[str, Any],
        criteria: dict[str, Any],
        product_type: str | None,
    ) -> dict[str, Any]:
        merged = dict(criteria)
        if product_type:
            merged["product_type"] = product_type
            merged["product_types"] = [product_type]
        merged["category_code"] = (
            context.get("category_code") or merged.get("category_code") or "Fashion"
        )
        merged["catalog_coverage"] = "supported"
        merged["catalog_available_product_types"] = self._available_product_types()

        for field in ["gender", "use_case", "color", "size", "style"]:
            if merged.get(field) is None and context.get(field) is not None:
                merged[field] = context.get(field)
        self._apply_price_context(merged, context)

        keywords = list(merged.get("keywords") or [])
        if product_type and product_type not in keywords:
            keywords.append(product_type)
        for field in ["gender", "use_case", "color", "size", "style"]:
            value = merged.get(field)
            if value and value not in keywords:
                keywords.append(value)
        merged["keywords"] = keywords
        merged["missing_fields"] = self._missing_collect_fields(merged)
        return merged

    def _missing_collect_fields(self, criteria: dict[str, Any]) -> list[str]:
        missing = []
        for field in REQUIRED_COLLECT_FIELDS:
            if field == "budget_max_vnd":
                if (
                    criteria.get("budget_min_vnd") is None
                    and criteria.get("budget_max_vnd") is None
                ):
                    missing.append(field)
                continue
            if criteria.get(field) is None:
                missing.append(field)
        return missing

    def _has_collected_required_preferences(self, criteria: dict[str, Any]) -> bool:
        has_budget = (
            criteria.get("budget_min_vnd") is not None
            or criteria.get("budget_max_vnd") is not None
        )
        if not criteria.get("product_type") or not has_budget:
            return False
        has_audience_or_use = bool(criteria.get("gender") or criteria.get("use_case"))
        has_context_or_variant = bool(
            criteria.get("use_case") or criteria.get("color") or criteria.get("size")
        )
        return has_audience_or_use and has_context_or_variant

    def _preference_clarification_questions(
        self, criteria: dict[str, Any]
    ) -> list[str]:
        missing = set(self._missing_collect_fields(criteria))
        if not missing:
            return []

        product_type = str(criteria.get("product_type") or "sản phẩm này").lower()
        if missing == {"gender", "use_case", "budget_max_vnd"}:
            return [
                f"Bạn muốn tìm {product_type} cho nam/nữ/unisex, mặc đi làm hay đi chơi, và ngân sách khoảng bao nhiêu ạ?"
            ]

        parts: list[str] = []
        if "gender" in missing:
            parts.append("cho nam/nữ/unisex")
        if "use_case" in missing:
            parts.append("mặc đi làm, đi chơi, đi học hay mặc hằng ngày")
        if "budget_max_vnd" in missing:
            parts.append("ngân sách khoảng bao nhiêu")

        if len(parts) == 1:
            question = f"Dạ, {parts[0]} ạ?"
        else:
            question = (
                "Dạ, bạn cho mình thêm " + ", ".join(parts[:-1]) + f" và {parts[-1]} ạ?"
            )
        return [question]

    def _attach_nearest_over_budget(
        self, criteria: dict[str, Any]
    ) -> list[dict[str, Any]]:
        budget = criteria.get("budget_max_vnd")
        if not budget:
            return []

        relaxed = dict(criteria)
        relaxed["budget_max_vnd"] = None
        relaxed["must_be_in_stock"] = True
        relaxed.pop("nearest_over_budget_products", None)
        relaxed.pop("rag_contexts", None)
        candidates = self.search_products(relaxed, top_k=50)
        over_budget = [
            product
            for product in candidates
            if self._matches_primary_budget_fallback_filters(product, criteria)
            and (resolve_product_price(product) or 0) > int(budget)
        ]
        if not over_budget:
            return []

        over_budget.sort(
            key=lambda product: (
                int(resolve_product_price(product) or 0) - int(budget),
                -int(product.get("stock_total") or 0),
            )
        )
        nearest = dict(over_budget[0])
        nearest_price = int(resolve_product_price(nearest) or 0)
        if nearest_price > int(int(budget) * NEAREST_OVER_BUDGET_MULTIPLIER):
            return []

        difference = nearest_price - int(budget)
        nearest["effective_price_vnd"] = nearest_price
        nearest["difference_vnd"] = difference
        nearest["budget_difference_vnd"] = difference
        nearest["over_budget_difference_vnd"] = difference
        criteria["response_mode"] = "nearest_over_budget_product"
        criteria["nearest_over_budget_products"] = [nearest]
        return [nearest]

    def _matches_primary_budget_fallback_filters(
        self,
        product: dict[str, Any],
        criteria: dict[str, Any],
    ) -> bool:
        if self._product_stock(product) <= 0:
            return False

        requested_type = criteria.get("product_type")
        product_type = product.get("product_type")
        if (
            requested_type
            and product_type
            and str(product_type).lower() != str(requested_type).lower()
        ):
            return False

        requested_types = {
            str(item).lower() for item in criteria.get("product_types") or [] if item
        }
        if (
            requested_types
            and product_type
            and str(product_type).lower() not in requested_types
        ):
            return False

        gender = criteria.get("gender")
        if gender and not self._product_matches_gender(product, str(gender)):
            return False

        for field in ["use_case", "style"]:
            value = criteria.get(field)
            if value and not self._product_matches_text(product, str(value)):
                return False

        return True

    def _product_matches_gender(self, product: dict[str, Any], gender: str) -> bool:
        gender_key = strip_accents(normalize_text(gender))
        if not gender_key:
            return True
        if self._product_matches_text(product, gender):
            return True

        searchable = strip_accents(
            normalize_text(
                " ".join(
                    str(product.get(field) or "")
                    for field in [
                        "product_type",
                        "tags",
                        "short_description",
                        "search_text",
                    ]
                )
            )
        )
        if gender_key in {"nam", "nu"} and "unisex" in searchable:
            return True
        if gender_key == "nu" and str(product.get("product_type") or "") in {
            "Váy",
            "Chân váy",
        }:
            return True
        return False

    def _product_matches_text(self, product: dict[str, Any], value: str) -> bool:
        value_key = strip_accents(normalize_text(value))
        if not value_key:
            return True
        searchable = strip_accents(
            normalize_text(
                " ".join(
                    str(product.get(field) or "")
                    for field in [
                        "product_name",
                        "product_type",
                        "tags",
                        "short_description",
                        "search_text",
                    ]
                )
            )
        )
        return value_key in searchable

    def _product_stock(self, product: dict[str, Any]) -> int:
        for field in ["stock_total", "available_qty", "quantity", "stock"]:
            value = product.get(field)
            if value is None or value == "":
                continue
            try:
                return int(float(value or 0))
            except (TypeError, ValueError):
                continue
        return 0

    def _product_is_available_for_empty_search(self, product: dict[str, Any]) -> bool:
        if self._product_stock(product) <= 0:
            return False
        for field in ["inventory_status", "status", "listing_status"]:
            status = strip_accents(normalize_text(str(product.get(field) or ""))).lower()
            status = status.replace("_", " ").replace("-", " ")
            if status in {"out of stock", "sold out", "het hang"}:
                return False
        return True

    def _handle_structured_availability(
        self, criteria: dict[str, Any]
    ) -> list[dict[str, Any]]:
        if criteria.get("intent") == "out_of_scope_request":
            return []

        if criteria.get("selected_product_id") or criteria.get(
            "requested_product_name"
        ):
            availability = self.search_engine.check_variant_availability(criteria)
        else:
            availability = self.search_engine.check_availability(criteria)
        criteria["availability"] = availability
        if availability.get("status") == "database_error":
            criteria["product_data_error"] = (
                availability.get("error") or "database_error"
            )
            criteria["response_mode"] = "product_data_error"
            return []
        if availability.get("status") != "in_stock":
            if availability.get("status") == "variant_options":
                return list(availability.get("alternatives") or [])
            return []

        product = dict(availability.get("product") or {})
        variant = availability.get("variant") or {}
        product.update(
            {
                "sku": variant.get("sku"),
                "size": variant.get("size"),
                "color": variant.get("color"),
                "stock": variant.get("stock"),
                "variant_status": variant.get("status"),
            }
        )
        return [product]

    def _handle_size_recommendation(
        self,
        message: str,
        top_k: int,
        conversation_id: str,
        criteria: dict[str, Any],
    ) -> dict[str, Any]:
        criteria = dict(criteria)
        context = CONVERSATION_STORE.get(conversation_id) or {}
        criteria.pop("size_recommendation", None)
        criteria.pop("recommended_size", None)
        criteria.pop("alternative_size", None)
        context_product_type = context.get("last_product_type") or context.get(
            "product_type"
        )

        if not criteria.get("product_type") and context_product_type:
            criteria["product_type"] = context_product_type
            criteria["product_types"] = [context_product_type]
        if criteria.get("product_type") and not criteria.get("category_code"):
            criteria["category_code"] = context.get("category_code", "Fashion")
        if not criteria.get("gender") and context.get("gender"):
            criteria["gender"] = context.get("gender")
        if not criteria.get("use_case") and context.get("use_case"):
            criteria["use_case"] = context.get("use_case")
        if not criteria.get("fit_preference") and context.get("fit_preference"):
            criteria["fit_preference"] = context.get("fit_preference")
        if not criteria.get("height_cm") and context.get("height_cm"):
            criteria["height_cm"] = context.get("height_cm")
        if not criteria.get("weight_kg") and context.get("weight_kg"):
            criteria["weight_kg"] = context.get("weight_kg")

        criteria["intent"] = "size_recommendation"
        criteria["catalog_coverage"] = "supported"
        criteria["catalog_available_product_types"] = self._available_product_types(
            "Fashion"
        )
        criteria.setdefault(
            "product_types",
            [criteria["product_type"]] if criteria.get("product_type") else [],
        )

        if not criteria.get("product_type"):
            criteria.update(
                {
                    "response_mode": "ask_product_for_size",
                    "need_clarification": True,
                    "clarification_questions": [
                        "Bạn đang muốn chọn size cho áo thun, áo sơ mi hay sản phẩm nào ạ?"
                    ],
                    "rag_contexts": [],
                    "rag_status": self.rag_service.status(),
                }
            )
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        if not criteria.get("height_cm") or not criteria.get("weight_kg"):
            criteria.update(
                {
                    "response_mode": "ask_size_profile",
                    "need_clarification": True,
                    "clarification_questions": [
                        "Bạn cho mình chiều cao và cân nặng để mình tư vấn size theo bảng size của shop ạ."
                    ],
                    "rag_contexts": [],
                    "rag_status": self.rag_service.status(),
                }
            )
            pending_action = self._store_pending_action(conversation_id, criteria, [])
            return self._chat_payload(
                conversation_id=conversation_id,
                pending_action=pending_action,
                criteria=criteria,
                reply=build_reply(criteria, []),
                products=[],
                rag_contexts=[],
            )

        recommendation = recommend_size(
            product_type=str(criteria["product_type"]),
            gender=criteria.get("gender"),
            height_cm=int(criteria["height_cm"]),
            weight_kg=int(criteria["weight_kg"]),
            fit_preference=criteria.get("fit_preference"),
        )
        criteria["fit_preference"] = recommendation.get(
            "fit_preference"
        ) or criteria.get("fit_preference")
        needs_fit_choice = bool(
            recommendation.get("boundary_case")
        ) and not criteria.get("fit_preference")
        criteria.update(
            {
                "response_mode": "size_recommendation",
                "need_clarification": needs_fit_choice,
                "clarification_questions": (
                    ["Bạn thích mặc ôm, vừa hay rộng ạ?"] if needs_fit_choice else []
                ),
                "size_recommendation": recommendation,
            }
        )
        rag_contexts = self._attach_rag_contexts(message, criteria, [], top_k)
        reply = build_reply(criteria, [])
        pending_action = self._store_pending_action(conversation_id, criteria, [])
        return self._chat_payload(
            conversation_id=conversation_id,
            pending_action=pending_action,
            criteria=criteria,
            reply=reply,
            products=[],
            rag_contexts=rag_contexts,
        )

    def _attach_rag_contexts(
        self,
        message: str,
        criteria: dict[str, Any],
        products: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        if self.catalog_source == "database":
            criteria["rag_contexts"] = []
            criteria["rag_status"] = {
                **self.rag_service.status(),
                "skipped_for_database_catalog": True,
            }
            return []

        rag_contexts = self.rag_service.retrieve_contexts(
            query=message,
            criteria=criteria,
            products=products,
            top_k=max(5, top_k),
        )
        criteria["rag_contexts"] = rag_contexts
        criteria["rag_status"] = self.rag_service.status()
        return rag_contexts

    def _rerank_products_with_rag(
        self,
        products: list[dict[str, Any]],
        rag_contexts: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        if not products or not rag_contexts:
            return products

        rag_score_by_product: dict[str, float] = {}
        for context in rag_contexts:
            product_id = context.get("product_id")
            score = context.get("score")
            if not product_id or score is None:
                continue
            rag_score_by_product[str(product_id)] = max(
                rag_score_by_product.get(str(product_id), 0.0),
                float(score),
            )

        if not rag_score_by_product:
            return products

        reranked: list[dict[str, Any]] = []
        for product in products:
            product = dict(product)
            rag_score = rag_score_by_product.get(str(product.get("product_id")))
            if rag_score:
                product["score"] = round(
                    float(product.get("score") or 0.0) + rag_score * 15, 3
                )
                reasons = list(product.get("matched_reasons") or [])
                if "khớp ngữ cảnh RAG" not in reasons:
                    reasons.append("khớp ngữ cảnh RAG")
                product["matched_reasons"] = reasons
            reranked.append(product)

        reranked.sort(key=lambda item: item.get("score", 0), reverse=True)
        return reranked

    def _canonical_fashion_product_type(self, text: str) -> str | None:
        return self.search_engine.catalog_gate.canonical_product_type(
            normalize_text(text)
        )

    def _available_product_types(self, category_code: str | None = None) -> list[str]:
        return self.search_engine.catalog_gate.dimensions.available_product_types(
            category_code
        )

    def conversation_memory(self, conversation_id: str) -> dict[str, Any]:
        return dict(CONVERSATION_STORE.get(conversation_id) or {})

    def reset_conversation(self, conversation_id: str) -> dict[str, bool]:
        conversation_cleared = conversation_id in CONVERSATION_STORE
        order_draft_cleared = order_draft_service.clear_draft(conversation_id)
        CONVERSATION_STORE.pop(conversation_id, None)
        return {
            "conversation_cleared": conversation_cleared,
            "order_draft_cleared": order_draft_cleared,
        }

    def _looks_like_size_request(self, text: str) -> bool:
        text_key = strip_accents(normalize_text(text))
        return bool(
            "size" in text_key
            or re_search_height_weight(text_key)
            or "tu van size" in text_key
            or "chon size" in text_key
        )

    def _chat_payload(
        self,
        *,
        conversation_id: str,
        pending_action: str | None,
        criteria: dict[str, Any],
        reply: str,
        products: list[dict[str, Any]],
        rag_contexts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if self.product_data_source == "postgres":
            criteria = dict(criteria)
            criteria["model_context"] = build_model_context(
                user_message=str(criteria.get("raw_message") or ""),
                product_data_from_database=products,
                rag_context=rag_contexts,
            )
        product_ids = [
            str(product.get("product_id"))
            for product in products
            if product.get("product_id")
        ]
        memory = self.conversation_memory(conversation_id)
        return {
            "conversation_id": conversation_id,
            "pending_action": pending_action,
            "criteria": criteria,
            "reply": reply,
            "products": products,
            "product_ids": product_ids,
            "rag_contexts": rag_contexts,
            "order_draft": memory.get("order_draft"),
            "conversation_memory": memory,
        }

    def _pending_action_for_criteria(self, criteria: dict[str, Any]) -> str | None:
        response_mode = criteria.get("response_mode")
        if response_mode == "ask_use_case":
            return "choose_use_case"
        if response_mode in {
            "ask_product_type",
            "choose_product_type",
            "invalid_product_type_choice",
        }:
            return "choose_product_type"
        if response_mode == "collect_product_preferences":
            return "collect_product_preferences"
        if response_mode == "ask_product_for_size":
            return "choose_product_for_size"
        if response_mode == "ask_size_profile":
            return "collect_size_profile"
        if response_mode == "ask_product_for_variant":
            return "choose_product_for_variant"
        if response_mode == "offer_closest_product_type":
            return "confirm_closest_product_type"
        if response_mode == "same_type_alternative_not_found":
            return "confirm_cross_type_alternative"
        if response_mode in {
            "product_type_out_of_stock",
            "product_type_not_found",
        }:
            return "browse_available_products"
        if (
            response_mode == "size_recommendation"
            and (criteria.get("size_recommendation") or {}).get("boundary_case")
            and not criteria.get("fit_preference")
        ):
            return "choose_fit_preference"
        return None

    def _store_pending_action(
        self,
        conversation_id: str,
        criteria: dict[str, Any],
        products: list[dict[str, Any]],
    ) -> str | None:
        pending_action = self._pending_action_for_criteria(criteria)
        existing = dict(CONVERSATION_STORE.get(conversation_id) or {})
        if criteria.get("reset_product_context"):
            existing = {}
        product_ids = [
            str(product.get("product_id"))
            for product in products
            if product.get("product_id")
        ]
        has_profile = any(
            criteria.get(field) is not None
            for field in [
                "product_type",
                "gender",
                "use_case",
                "style",
                "color",
                "budget_min_vnd",
                "budget_max_vnd",
                "size",
                "height_cm",
                "weight_kg",
                "fit_preference",
                "requested_product_group",
                "suggested_product_type",
            ]
        )

        if not pending_action and not product_ids and not has_profile and not existing:
            CONVERSATION_STORE.pop(conversation_id, None)
            return None

        memory = existing
        for field, default_value in MEMORY_DEFAULTS.items():
            if field not in memory:
                memory[field] = (
                    list(default_value)
                    if isinstance(default_value, list)
                    else default_value
                )

        memory["pending_action"] = pending_action
        if criteria.get("intent"):
            memory["intent"] = criteria.get("intent")
        if criteria.get("response_mode"):
            memory["response_mode"] = criteria.get("response_mode")
        if criteria.get("category_code"):
            memory["category_code"] = criteria.get("category_code")
        if criteria.get("catalog_available_product_types"):
            memory["available_product_types"] = criteria.get(
                "catalog_available_product_types"
            )
        if criteria.get("response_mode") == "offer_closest_product_type":
            memory.pop("last_product_type", None)
            memory.pop("product_type", None)
            memory.pop("last_product_ids", None)
            memory["selected_product_id"] = None
            memory["selected_variant_sku"] = None
        elif (
            criteria.get("response_mode") == "cross_type_alternative"
            and criteria.get("product_type") is None
        ):
            memory.pop("last_product_type", None)
            memory.pop("product_type", None)
            memory["selected_product_id"] = None
            memory["selected_variant_sku"] = None
        elif (
            criteria.get("response_mode") == "catalog_browsing"
            and not criteria.get("product_type")
        ):
            memory.pop("last_product_type", None)
            memory.pop("product_type", None)
        elif (
            not pending_action
            and not product_ids
            and criteria.get("catalog_coverage") in {"unsupported", "not_supported"}
        ):
            CONVERSATION_STORE.pop(conversation_id, None)
            return None
        if criteria.get("product_type"):
            memory["last_product_type"] = criteria.get("product_type")
            memory["product_type"] = criteria.get("product_type")
        if product_ids:
            memory["last_product_ids"] = product_ids
            memory["selected_product_id"] = product_ids[0]
        elif criteria.get("intent") in {
            "reject_current_product",
            "request_alternative_product",
            "request_similar_product",
        } and criteria.get("nearest_over_budget_products"):
            nearest_ids = [
                str(product.get("product_id"))
                for product in criteria.get("nearest_over_budget_products") or []
                if product.get("product_id")
            ]
            if nearest_ids:
                memory["last_product_ids"] = nearest_ids
                memory["selected_product_id"] = nearest_ids[0]
        elif criteria.get("product_type") and "last_product_ids" not in memory:
            memory["last_product_ids"] = []
        for field in [
            "gender",
            "use_case",
            "style",
            "color",
            "budget_min_vnd",
            "budget_max_vnd",
            "price_direction",
            "size",
            "height_cm",
            "weight_kg",
            "fit_preference",
            "requested_product_group",
            "suggested_product_type",
        ]:
            if criteria.get(field) is not None:
                memory[field] = criteria.get(field)
        if self._has_explicit_price_filter(criteria):
            if (
                criteria.get("budget_min_vnd") is not None
                and criteria.get("budget_max_vnd") is None
                and criteria.get("price_direction") != "between"
            ):
                memory.pop("budget_max_vnd", None)
            if (
                criteria.get("budget_max_vnd") is not None
                and criteria.get("budget_min_vnd") is None
                and criteria.get("price_direction") != "between"
            ):
                memory.pop("budget_min_vnd", None)
        if (
            criteria.get("style") is None
            and criteria.get("use_case")
            and memory.get("style")
            and normalize_text(str(memory["style"]))
            == normalize_text(str(criteria["use_case"]))
        ):
            memory.pop("style", None)
        if criteria.get("missing_fields"):
            memory["missing_fields"] = criteria.get("missing_fields")
        if criteria.get("size_recommendation"):
            recommendation = criteria.get("size_recommendation")
            memory["size_recommendation"] = recommendation
            memory["recommended_size"] = recommendation.get("recommended_size")
            memory["alternative_size"] = recommendation.get("alternative_size")

        if criteria.get("selected_product_id"):
            selected_candidate = str(criteria.get("selected_product_id"))
            rejected_in_criteria = {
                str(product_id)
                for product_id in criteria.get("rejected_product_ids") or []
                if product_id
            }
            if not (
                criteria.get("intent")
                in {
                    "reject_current_product",
                    "request_alternative_product",
                    "request_similar_product",
                }
                and selected_candidate in rejected_in_criteria
            ):
                memory["selected_product_id"] = selected_candidate

        availability = criteria.get("availability") or {}
        variant = availability.get("variant") or {}
        if variant.get("sku"):
            memory["selected_variant_sku"] = variant.get("sku")
        elif products:
            first_sku = products[0].get("sku")
            if first_sku:
                memory["selected_variant_sku"] = first_sku

        rejected_product_ids = [
            str(product_id)
            for product_id in [
                *(memory.get("rejected_product_ids") or []),
                *(criteria.get("rejected_product_ids") or []),
            ]
            if product_id
        ]
        memory["rejected_product_ids"] = list(dict.fromkeys(rejected_product_ids))

        CONVERSATION_STORE[conversation_id] = memory
        return pending_action

    def _handle_order_size_recommendation(
        self,
        conversation_id: str,
        context: dict[str, Any],
        criteria: dict[str, Any],
        message: str,
    ) -> dict[str, Any]:
        draft = ensure_order_draft(context)
        item = first_item(draft)

        product_type = context.get("product_type") or context.get("last_product_type")
        gender = criteria.get("gender") or context.get("gender")

        if not product_type:
            return self._order_response(
                conversation_id,
                context,
                context.get("pending_action"),
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "ask_product_for_size",
                },
                "Bạn đang muốn chọn size cho sản phẩm nào ạ?",
            )

        recommendation = recommend_size(
            product_type=str(product_type),
            gender=gender,
            height_cm=int(criteria["height_cm"]),
            weight_kg=int(criteria["weight_kg"]),
            fit_preference=(
                criteria.get("fit_preference") or context.get("fit_preference")
            ),
        )

        recommended_size = recommendation.get("recommended_size")
        alternative_size = recommendation.get("alternative_size")

        context["height_cm"] = criteria["height_cm"]
        context["weight_kg"] = criteria["weight_kg"]
        context["recommended_size"] = recommended_size
        context["alternative_size"] = alternative_size

        if not recommended_size:
            return self._order_response(
                conversation_id,
                context,
                "choose_order_size",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "size_recommendation_unavailable",
                    "size_recommendation": recommendation,
                },
                recommendation.get("reason")
                or "Mình chưa có bảng size phù hợp cho sản phẩm này.",
            )

        availability = self.search_engine.check_variant_availability(
            {
                "selected_product_id": item.get("product_id"),
                "color": item.get("color"),
                "size": recommended_size,
                "shop_id": self.active_shop_id,
            }
        )

        if availability.get("status") == "in_stock":
            set_item_variant(
                item,
                availability["variant"],
                availability.get("product"),
            )
            item["unit_price"] = (availability.get("product") or {}).get(
                "effective_price_vnd"
            ) or item.get("unit_price")

            reply = (
                f"Dựa trên chiều cao {criteria['height_cm']}cm và "
                f"cân nặng {criteria['weight_kg']}kg, mình gợi ý size "
                f"{recommended_size}. {recommendation.get('reason', '')}\n\n"
                f"Mẫu {item.get('product_name')} size {recommended_size} "
                f"hiện còn {item.get('variant_stock')} sản phẩm. "
                "Bạn muốn lấy bao nhiêu cái ạ?"
            )

            return self._order_response(
                conversation_id,
                context,
                "choose_order_quantity",
                {
                    **criteria,
                    "intent": "order_draft",
                    "response_mode": "order_size_recommended",
                    "size_recommendation": recommendation,
                    "availability": availability,
                },
                reply,
            )

        alternatives = availability.get("alternatives") or []

        option_lines = []
        for alternative in alternatives[:5]:
            option_lines.append(
                f"- Màu {alternative.get('color')}, size {alternative.get('size')}"
            )

        reply = (
            f"Dựa trên chiều cao {criteria['height_cm']}cm và "
            f"cân nặng {criteria['weight_kg']}kg, mình gợi ý size "
            f"{recommended_size}. {recommendation.get('reason', '')}\n\n"
            f"Tuy nhiên mẫu {item.get('product_name')} hiện chưa có size "
            f"{recommended_size}."
        )

        if option_lines:
            reply += "\nShop hiện còn:\n" + "\n".join(option_lines)
            reply += "\nBạn muốn chọn size nào ạ?"
        else:
            reply += "\nHiện shop chưa có size phù hợp cho mẫu này."

        return self._order_response(
            conversation_id,
            context,
            "choose_order_size",
            {
                **criteria,
                "intent": "order_draft",
                "response_mode": "recommended_size_unavailable",
                "size_recommendation": recommendation,
                "availability": availability,
            },
            reply,
        )


def re_search_height_weight(text: str) -> bool:
    return bool(
        re.search(r"\b\d\s*m\s*\d{1,2}\b", text)
        or re.search(r"\b\d{2,3}\s*cm\b", text)
        or re.search(r"\b\d{2,3}\s*kg\b", text)
    )


@lru_cache(maxsize=1)
def get_service() -> ChatbotService:
    return ChatbotService()
