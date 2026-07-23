from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd

sys.dont_write_bytecode = True
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ["PRODUCT_DATA_SOURCE"] = "csv"
os.environ.setdefault("SHOP_ID", "00000000-0000-0000-0000-000000000001")

try:
    import pytest
except ImportError:  # pragma: no cover - manual runner can work without pytest.
    pytest = None

from app.rag_retriever import AutoBizRAGRetriever  # noqa: E402
import app.service as service_module  # noqa: E402
from app.order_draft_service import SubmissionResult, build_n8n_payload  # noqa: E402
from app.search_engine import ProductSearchEngine  # noqa: E402
from app.service import CONVERSATION_STORE, ChatbotService  # noqa: E402
from app.size_recommender import recommend_size  # noqa: E402

if pytest is not None:

    @pytest.fixture(scope="module")
    def service() -> ChatbotService:
        CONVERSATION_STORE.clear()
        return ChatbotService()


CATALOG_OVERVIEW_QUESTIONS = [
    "shop bán những sản phẩm gì?",
    "shop có những sản phẩm gì?",
    "bên mình bán gì?",
    "shop đang bán gì?",
    "shop có bán gì không?",
    "cửa hàng có những mặt hàng nào?",
    "shop có những nhóm sản phẩm nào?",
    "shop bán những loại hàng nào?",
    "shop mình có gì?",
]

QUESTIONS = [
    "shop bán những sản phẩm gì?",
    "Shop mình có bán mấy sản phẩm quần áo đá bóng không",
    "SHOP CÓ BỘ QUẦN ÁO ĐÁ BÓNG CỦA ĐỘI TUYỂN BỒ ĐÀO NHA K?",
    "Shop có áo bóng đá Bồ Đào Nha không?",
    "Shop có áo khoác da biker không?",
    "Shop còn Áo thun basic unisex màu mới size M màu sọc xanh trắng không?",
    "Áo sơ mi Oxford best seller size M màu đỏ đô còn không?",
    "Tôi muốn mua Áo sơ mi denim mỏng mùa hè size L màu xanh navy và thêm giày sneaker trắng, shop có không?",
    "Shop có bán bánh mì và nước cam không?",
    "Tôi cần mua một cái áo ba lỗ dành cho nam màu trắng để mặc thoải mái ở nhà",
    "Mình cần áo sơ mi nữ đi làm màu trắng, giá dưới 300k, còn size M không?",
    "Mình cần áo sơ mi nữ đi làm màu đen, giá dưới 300k, còn size M không?",
    "Da mình dầu và hay nổi mụn, tư vấn sữa rửa mặt hoặc serum dưới 250k",
    "Có đồ ăn healthy ít calo cho buổi trưa không?",
    "Shop có bỉm hoặc đồ chơi em bé không?",
    "Shop có bán giày sneaker trắng không?",
    "Shop có bán túi xách không?",
    "Shop có bán nước hoa không?",
    "Shop có bán mỹ phẩm không?",
    "Tìm quà sinh nhật cho bạn nữ, phụ kiện style vintage hoặc cute",
    "Có váy nào đẹp không?",
    "shop bán những bộ váy như nào?",
    "shop có áo không",
    "gợi ý đồ đi chơi",
]

MANUAL_PRODUCT_DATA_SOURCE_CHECKS = [
    "Áo thun basic màu trắng size S còn không?",
    "Giá áo blazer công sở là bao nhiêu?",
    "Có sản phẩm nào dưới 300 nghìn không?",
    "Sản phẩm nào sắp hết hàng?",
]


def validate_result(question: str, result: dict) -> None:
    criteria = result["criteria"]
    products = result["products"]
    reply = result["reply"].lower()
    gate = criteria.get("catalog_gate") or {}

    assert all(
        product.get("category_code") in {"Fashion", "Accessories"} for product in products
    ), products

    if criteria.get("catalog_coverage") == "unsupported":
        assert gate.get("status") == "unsupported", criteria
        assert criteria.get("category_code") is None, criteria
        assert criteria.get("product_type") is None, criteria
        assert criteria.get("product_types") == [], criteria
        assert criteria.get("apparel_intent") is None, criteria
        assert products == [], products

    if "sữa rửa mặt hoặc serum" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert "không có sản phẩm này" in reply and "sữa rửa mặt" in reply and "serum" in reply, result["reply"]
        assert "áo thun" in reply and "áo sơ mi" in reply, result["reply"]

    if "bánh mì và nước cam" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert "không có sản phẩm này" in reply and "bánh mì" in reply and "nước cam" in reply, result["reply"]
        assert "áo thun" in reply and "váy" in reply, result["reply"]

    if "áo ba lỗ" in question:
        assert criteria.get("intent") == "unsupported_product_request", criteria
        assert criteria.get("response_mode") == "offer_closest_product_type", criteria
        assert criteria.get("requested_product_group") == "áo ba lỗ", criteria
        assert criteria.get("suggested_product_type") == "Áo thun", criteria
        assert criteria.get("product_type") is None, criteria
        assert criteria.get("product_types") == [], criteria
        assert criteria.get("catalog_coverage") == "closest_alternative_available", criteria
        assert products == [], products
        assert result["product_ids"] == [], result
        assert result["pending_action"] == "confirm_closest_product_type", result
        assert "áo thun" in reply, result["reply"]

    if "healthy ít calo" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert "không có sản phẩm này" in reply and "đồ ăn" in reply, result["reply"]
        assert "áo thun" in reply and "váy" in reply, result["reply"]

    if "bỉm hoặc đồ chơi em bé" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert "không có sản phẩm này" in reply and "bỉm" in reply and "đồ chơi" in reply, result["reply"]
        assert "áo thun" in reply and "set bộ" in reply, result["reply"]

    if "giày sneaker trắng" in question and "denim mỏng mùa hè" not in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert criteria.get("out_of_scope_items") == ["giày sneaker"], criteria
        assert "không có sản phẩm này" in reply and "giày sneaker" in reply, result["reply"]

    if "túi xách" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert criteria.get("out_of_scope_items") == ["túi xách"], criteria
        assert "không có sản phẩm này" in reply and "túi xách" in reply, result["reply"]

    if "nước hoa" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert "nước hoa" in criteria.get("out_of_scope_items", []), criteria
        assert "không có sản phẩm này" in reply and "nước hoa" in reply, result["reply"]

    if question == "Shop có bán mỹ phẩm không?":
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert "không có sản phẩm này" in reply and "mỹ phẩm" in reply, result["reply"]

    if "quần áo đá bóng" in question:
        assert criteria.get("intent") == "out_of_scope_request", criteria
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert products == [], products
        assert result.get("pending_action") is None, result
        assert "không có sản phẩm này" in reply and "quần áo đá bóng" in reply, result["reply"]
        assert "gợi ý" not in reply, result["reply"]

    if "đội tuyển bồ đào nha" in question.lower():
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert criteria.get("out_of_scope_items") == ["bộ quần áo đá bóng", "đội tuyển bồ đào nha"], criteria
        assert result["product_ids"] == [], result

    if "áo bóng đá" in question.lower():
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert criteria.get("out_of_scope_items") == ["áo bóng đá", "bồ đào nha"], criteria
        assert result["product_ids"] == [], result

    if "áo khoác da biker" in question.lower():
        assert criteria.get("catalog_coverage") == "unsupported", criteria
        assert "áo khoác da biker" in criteria.get("out_of_scope_items", []), criteria
        assert result["product_ids"] == [], result

    if "Áo thun basic unisex màu mới" in question:
        assert criteria.get("intent") == "product_availability_check", criteria
        assert criteria.get("requested_product_name") == "Áo thun basic unisex màu mới", criteria
        assert products and products[0]["product_id"] == "P0002", products
        assert products[0].get("sku") == "ABF-0002-M-613", products
        assert products[0].get("stock") == 34, products
        assert "còn hàng" in reply and "167.000đ" in result["reply"], result["reply"]

    if "Oxford best seller" in question:
        assert criteria.get("intent") == "product_availability_check", criteria
        assert criteria.get("availability", {}).get("status") == "out_of_stock", criteria
        assert products == [], products
        assert "hết hàng" in reply and "trắng" in reply and "xanh nhạt" in reply, result["reply"]

    if "denim mỏng mùa hè" in question:
        assert criteria.get("intent") == "mixed_product_request", criteria
        assert criteria.get("catalog_coverage") == "partially_supported", criteria
        assert criteria.get("out_of_scope_items") == ["giày sneaker"], criteria
        assert products and products[0]["product_id"] == "P0033", products
        assert products[0].get("sku") == "ABF-0033-L-367", products
        assert "shop không bán giày sneaker" in reply, result["reply"]

    if "màu đen" in question and "áo sơ mi" in question:
        assert criteria.get("catalog_coverage") == "supported", criteria
        assert criteria.get("product_type") == "Áo sơ mi", criteria
        assert products and all(product["product_type"] == "Áo sơ mi" for product in products), products
        assert any(
            "chưa khớp chính xác màu/size" in reason
            for product in products
            for reason in product.get("matched_reasons", [])
        ), products

    if "shop bán những bộ váy như nào" in question:
        assert criteria.get("response_mode") == "catalog_browsing", criteria
        assert criteria.get("catalog_coverage") == "supported", criteria
        assert criteria.get("product_type") == "Váy", criteria
        assert criteria.get("need_clarification") is False, criteria
        assert products and all(product["product_type"] == "Váy" for product in products), products
        assert result["reply"].startswith("Dạ shop mình hiện có một số mẫu váy như:"), result["reply"]
        assert "mình tìm được vài sản phẩm phù hợp cho bạn" not in reply, result["reply"]
        assert "phù hợp với bạn" not in reply, result["reply"]
        assert "Váy Luna House Daily" in result["reply"], result["reply"]
        assert "Váy Nami Plus" in result["reply"], result["reply"]
        assert "còn hàng" in reply, result["reply"]
        assert "Bạn muốn mình lọc thêm theo size, màu hoặc ngân sách không ạ?" in result["reply"], result["reply"]

    if question == "shop có áo không":
        assert criteria.get("response_mode") == "catalog_browsing", criteria
        assert criteria.get("catalog_coverage") == "supported", criteria
        assert criteria.get("apparel_intent") == "áo", criteria
        assert criteria.get("need_clarification") is False, criteria
        assert products and all(product["product_type"].startswith("Áo") for product in products), products
        assert result["reply"].startswith("Dạ shop mình hiện có một số mẫu áo như:"), result["reply"]
        assert "mình tìm được vài sản phẩm phù hợp cho bạn" not in reply, result["reply"]
        assert "phù hợp với bạn" not in reply, result["reply"]

    if question in CATALOG_OVERVIEW_QUESTIONS:
        assert criteria.get("intent") == "catalog_overview", criteria
        assert criteria.get("response_mode") == "catalog_overview", criteria
        assert criteria.get("catalog_coverage") == "supported", criteria
        assert gate.get("status") == "supported", criteria
        assert criteria.get("requested_product_group") is None, criteria
        assert criteria.get("out_of_scope_items") == [], criteria
        assert gate.get("unsupported_terms") == [], criteria
        assert gate.get("requested_product_concepts") == [], criteria
        assert result["product_ids"] == [], result
        assert "không có sản phẩm này" not in reply, result["reply"]
        assert result["reply"].startswith("Dạ shop mình hiện đang bán các nhóm sản phẩm như:"), result["reply"]
        for product_type in ["Áo thun", "Áo sơ mi", "Váy", "Set bộ", "Mũ", "Khuyên tai"]:
            assert f"- {product_type}".lower() in reply, result["reply"]
        assert "Bạn muốn mình gợi ý theo nhu cầu nào" in result["reply"], result["reply"]


def print_result(title: str, message: str, result: dict) -> None:
    print("=" * 100)
    print(title)
    print("USER")
    print(message)
    print("CONVERSATION ID")
    print(result.get("conversation_id"))
    print("PENDING ACTION")
    print(result.get("pending_action"))
    print("CRITERIA")
    print(json.dumps(result["criteria"], ensure_ascii=False, indent=2))
    print("BOT reply")
    print(result["reply"])
    print("PRODUCT IDS")
    print(result["product_ids"])
    print("RAG CONTEXTS")
    print(json.dumps(result.get("rag_contexts", []), ensure_ascii=False, indent=2))


def run_conversation_flow(service: ChatbotService) -> None:
    conversation_id = "test_football"
    CONVERSATION_STORE.pop(conversation_id, None)

    unsupported_message = "Shop mình có bán mấy sản phẩm quần áo đá bóng không"
    unsupported = service.chat(unsupported_message, top_k=5, conversation_id=conversation_id)
    validate_result(unsupported_message, unsupported)
    assert unsupported["pending_action"] is None, unsupported
    assert conversation_id not in CONVERSATION_STORE, CONVERSATION_STORE
    print_result("FLOW 1: UNSUPPORTED PRODUCT GROUP", unsupported_message, unsupported)

    yes_message = "Có"
    yes_result = service.chat(yes_message, top_k=5, conversation_id=conversation_id)
    assert yes_result["products"] == [], yes_result
    assert yes_result["product_ids"] == [], yes_result
    yes_reply = yes_result["reply"].lower()
    assert "bạn muốn mình gợi ý sản phẩm thuộc nhóm nào" in yes_reply, yes_result["reply"]
    assert yes_result.get("pending_action") == "choose_product_type", yes_result
    print_result("FLOW 2: FOLLOW-UP YES WITHOUT UNSUPPORTED CONTEXT", yes_message, yes_result)

    no_context_message = "Có"
    no_context = service.chat(no_context_message, top_k=5, conversation_id="test_choose_type")
    assert no_context["products"] == [], no_context
    assert no_context["product_ids"] == [], no_context
    assert "bạn muốn mình gợi ý sản phẩm thuộc nhóm nào" in no_context["reply"].lower(), no_context["reply"]
    assert no_context["criteria"].get("intent") == "conversation_followup", no_context
    assert no_context["criteria"].get("response_mode") == "choose_product_type", no_context
    assert no_context["pending_action"] == "choose_product_type", no_context
    for product_type in ["áo thun", "váy", "áo sơ mi"]:
        assert product_type in no_context["reply"].lower(), no_context["reply"]
    print_result("FLOW 3: YES WITHOUT CONTEXT", no_context_message, no_context)


def run_choose_product_type_flow(service: ChatbotService) -> None:
    conversation_id = "choose_flow_1"
    CONVERSATION_STORE.pop(conversation_id, None)

    yes_message = "Có"
    yes_result = service.chat(yes_message, top_k=5, conversation_id=conversation_id)
    assert yes_result["products"] == [], yes_result
    assert yes_result["pending_action"] == "choose_product_type", yes_result
    yes_reply = yes_result["reply"].lower()
    assert "bạn muốn mình gợi ý sản phẩm thuộc nhóm nào" in yes_reply, yes_result["reply"]
    for product_type in [
        "áo thun",
        "áo sơ mi",
        "váy",
        "set bộ",
    ]:
        assert product_type in yes_reply, yes_result["reply"]
    assert "bạn chỉ cần nhắn tên nhóm sản phẩm" in yes_reply, yes_result["reply"]
    print_result("FLOW A: YES WITHOUT CONTEXT -> CHOOSE PRODUCT TYPE", yes_message, yes_result)

    choose_message = "áo thun"
    choose_result = service.chat(choose_message, top_k=5, conversation_id=conversation_id)
    assert choose_result["products"] == [], choose_result
    assert choose_result["pending_action"] == "collect_product_preferences", choose_result
    assert choose_result["criteria"].get("product_type") == "Áo thun", choose_result
    choose_reply = choose_result["reply"].lower()
    assert "nam/nữ/unisex" in choose_reply and "ngân sách" in choose_reply, choose_result["reply"]
    print_result("FLOW B: CHOOSE ÁO THUN", choose_message, choose_result)

    criteria_message = "nam màu trắng dưới 300k"
    criteria_result = service.chat(criteria_message, top_k=5, conversation_id=conversation_id)
    criteria = criteria_result["criteria"]
    assert criteria.get("category_code") == "Fashion", criteria
    assert criteria.get("product_type") == "Áo thun", criteria
    assert criteria.get("gender") == "nam", criteria
    assert criteria.get("color") == "trắng", criteria
    assert criteria.get("budget_max_vnd") == 300000, criteria
    assert criteria_result["pending_action"] is None, criteria_result
    for product in criteria_result["products"]:
        assert product["product_type"] == "Áo thun", product
        assert product["effective_price_vnd"] <= 300000, product
    if not criteria_result["products"]:
        assert "ngân sách 300.000đ" in criteria_result["reply"], criteria_result["reply"]
    print_result("FLOW C: COLLECT PREFERENCES AND SEARCH", criteria_message, criteria_result)


def run_invalid_product_type_flow(service: ChatbotService) -> None:
    conversation_id = "choose_flow_invalid"
    CONVERSATION_STORE.pop(conversation_id, None)

    service.chat("Có", top_k=5, conversation_id=conversation_id)
    invalid_message = "đồ đá bóng"
    invalid_result = service.chat(invalid_message, top_k=5, conversation_id=conversation_id)
    assert invalid_result["products"] == [], invalid_result
    assert invalid_result["pending_action"] == "choose_product_type", invalid_result
    invalid_reply = invalid_result["reply"].lower()
    assert "shop chưa có nhóm sản phẩm đó" in invalid_reply, invalid_result["reply"]
    assert "áo thun" in invalid_reply and "áo sơ mi" in invalid_reply, invalid_result["reply"]
    print_result("FLOW D: INVALID PRODUCT TYPE CHOICE", invalid_message, invalid_result)


def run_general_buying_and_size_cases(service: ChatbotService) -> None:
    general_conversation = "general_buying_case"
    CONVERSATION_STORE.pop(general_conversation, None)

    general_message = "tôi muốn mua đồ"
    general_result = service.chat(general_message, top_k=5, conversation_id=general_conversation)
    general_criteria = general_result["criteria"]
    general_reply = general_result["reply"].lower()
    assert general_result["products"] == [], general_result
    assert general_result["product_ids"] == [], general_result
    assert general_result["pending_action"] == "choose_product_type", general_result
    assert general_criteria.get("intent") == "general_buying_intent", general_criteria
    assert general_criteria.get("response_mode") == "ask_product_type", general_criteria
    assert general_criteria.get("need_clarification") is True, general_criteria
    assert "bạn đang muốn tìm sản phẩm nào" in general_reply, general_result["reply"]
    for product_type in ["áo thun", "áo sơ mi", "váy", "set bộ", "mũ", "khuyên tai"]:
        assert product_type in general_reply, general_result["reply"]
    assert "shop tìm được" not in general_reply, general_result["reply"]
    print_result("FLOW E: GENERAL BUYING INTENT", general_message, general_result)

    choose_message = "áo sơ mi"
    choose_result = service.chat(choose_message, top_k=5, conversation_id=general_conversation)
    choose_criteria = choose_result["criteria"]
    choose_reply = choose_result["reply"].lower()
    assert choose_result["products"] == [], choose_result
    assert choose_result["product_ids"] == [], choose_result
    assert choose_result["pending_action"] == "collect_product_preferences", choose_result
    assert choose_criteria.get("product_type") == "Áo sơ mi", choose_criteria
    assert "nam/nữ/unisex" in choose_reply, choose_result["reply"]
    assert "đi làm" in choose_reply and "ngân sách" in choose_reply, choose_result["reply"]
    print_result("FLOW F: GENERAL BUYING FOLLOW-UP PRODUCT TYPE", choose_message, choose_result)

    size_context_conversation = "size_with_context_case"
    CONVERSATION_STORE.pop(size_context_conversation, None)
    product_message = "Mình muốn mua áo sơ mi nam đi làm"
    product_result = service.chat(product_message, top_k=5, conversation_id=size_context_conversation)
    product_criteria = product_result["criteria"]
    assert product_criteria.get("product_type") == "Áo sơ mi", product_criteria
    assert product_criteria.get("gender") == "nam", product_criteria
    assert product_criteria.get("use_case") == "đi làm", product_criteria
    assert product_result["product_ids"], product_result
    assert product_result["conversation_memory"].get("last_product_type") == "Áo sơ mi", product_result
    assert product_result["conversation_memory"].get("last_product_ids"), product_result

    size_message = "mình cao 1m75 nặng 75kg thì nên mặc size nào?"
    size_result = service.chat(size_message, top_k=5, conversation_id=size_context_conversation)
    size_criteria = size_result["criteria"]
    size_reply = size_result["reply"].lower()
    assert size_result["products"] == [], size_result
    assert size_result["product_ids"] == [], size_result
    assert size_result["pending_action"] == "choose_fit_preference", size_result
    assert size_criteria.get("intent") == "size_recommendation", size_criteria
    assert size_criteria.get("product_type") == "Áo sơ mi", size_criteria
    assert size_criteria.get("gender") == "nam", size_criteria
    assert size_criteria.get("height_cm") == 175, size_criteria
    assert size_criteria.get("weight_kg") == 75, size_criteria
    assert size_criteria.get("size_recommendation", {}).get("recommended_size") in {"L", "XL"}, size_criteria
    assert "bạn muốn tìm nhóm sản phẩm nào" not in size_reply, size_result["reply"]
    assert "size l" in size_reply or "l/xl" in size_reply, size_result["reply"]
    assert "bảng size" not in size_reply, size_result["reply"]
    assert "ôm, vừa người hay rộng" in size_reply, size_result["reply"]
    print_result("FLOW G: SIZE RECOMMENDATION WITH PRODUCT CONTEXT", size_message, size_result)

    size_no_context_conversation = "size_without_context_case"
    CONVERSATION_STORE.pop(size_no_context_conversation, None)
    no_context_result = service.chat(size_message, top_k=5, conversation_id=size_no_context_conversation)
    no_context_criteria = no_context_result["criteria"]
    no_context_reply = no_context_result["reply"].lower()
    assert no_context_result["products"] == [], no_context_result
    assert no_context_result["product_ids"] == [], no_context_result
    assert no_context_result["pending_action"] == "choose_product_for_size", no_context_result
    assert no_context_criteria.get("intent") == "size_recommendation", no_context_criteria
    assert no_context_criteria.get("height_cm") == 175, no_context_criteria
    assert no_context_criteria.get("weight_kg") == 75, no_context_criteria
    assert "bạn đang muốn chọn size cho áo thun, áo sơ mi hay sản phẩm nào ạ" in no_context_reply, no_context_result["reply"]
    assert "shop tìm được" not in no_context_reply, no_context_result["reply"]
    print_result("FLOW H: SIZE RECOMMENDATION WITHOUT PRODUCT CONTEXT", size_message, no_context_result)


def run_size_recommender_boundary_cases(service: ChatbotService) -> None:
    overlap = recommend_size("Áo sơ mi", "nữ", 162, 55)
    assert overlap["recommended_size"] == "M", overlap
    assert overlap["alternative_size"] == "L", overlap
    assert overlap["boundary_case"] is True, overlap
    assert overlap["confidence"] == "medium", overlap
    assert "gần biên" in overlap["reason"], overlap
    assert set(overlap["fit_notes"]) == {"M", "L"}, overlap

    overlap_loose = recommend_size("Áo sơ mi", "nam", 175, 75, fit_preference="rộng")
    assert overlap_loose["recommended_size"] == "XL", overlap_loose
    assert overlap_loose["alternative_size"] == "L", overlap_loose
    assert overlap_loose["boundary_case"] is True, overlap_loose

    exact_boundary = recommend_size("Áo sơ mi", "nam", 170, 65, fit_preference="vừa")
    assert exact_boundary["recommended_size"] in {"M", "L"}, exact_boundary
    assert exact_boundary["alternative_size"] in {"M", "L"}, exact_boundary
    assert exact_boundary["recommended_size"] != exact_boundary["alternative_size"], exact_boundary
    assert exact_boundary["boundary_case"] is True, exact_boundary
    assert "vùng giao nhau" in exact_boundary["reason"], exact_boundary

    split_signal = recommend_size("Áo sơ mi", "nữ", 162, 62)
    assert split_signal["recommended_size"] == "L", split_signal
    assert split_signal["alternative_size"] == "M", split_signal
    assert split_signal["boundary_case"] is True, split_signal
    assert split_signal["confidence"] == "low", split_signal
    assert "không cùng rơi trọn một size" in split_signal["reason"], split_signal

    conversation_id = "size_boundary_reply_case"
    CONVERSATION_STORE.pop(conversation_id, None)
    service.chat("Mình muốn mua áo sơ mi nữ đi làm", top_k=5, conversation_id=conversation_id)
    reply_result = service.chat(
        "mình cao 1m62 nặng 55kg thì nên mặc size nào?",
        top_k=5,
        conversation_id=conversation_id,
    )
    reply = reply_result["reply"].lower()
    criteria = reply_result["criteria"]
    assert criteria.get("size_recommendation", {}).get("recommended_size") == "M", criteria
    assert criteria.get("size_recommendation", {}).get("alternative_size") == "L", criteria
    assert criteria.get("size_recommendation", {}).get("boundary_case") is True, criteria
    assert "bạn có thể chọn size m nếu thích mặc vừa người" in reply, reply_result["reply"]
    assert "có thể cân nhắc size l" in reply, reply_result["reply"]
    assert "ôm, vừa người hay rộng" in reply, reply_result["reply"]
    print_result(
        "FLOW I: SIZE BOUNDARY REPLY 162CM 55KG",
        "mình cao 1m62 nặng 55kg thì nên mặc size nào?",
        reply_result,
    )


def test_recommend_size_loose_prefers_larger_boundary_size() -> None:
    result = recommend_size(
        product_type="Áo sơ mi",
        gender="nữ",
        height_cm=162,
        weight_kg=55,
        fit_preference="loose",
    )

    assert result["recommended_size"] == "L", result
    assert result["alternative_size"] == "M", result
    assert result["fit_preference"] == "loose", result


def test_size_followup_loose_preference(service: ChatbotService) -> None:
    conversation_id = "test-size-loose-followup"
    CONVERSATION_STORE.pop(conversation_id, None)

    first = service.chat(
        "Tôi cao 1m62 nặng 55kg thì mặc size nào?",
        conversation_id=conversation_id,
    )

    assert first["pending_action"] == "choose_product_for_size", first

    second = service.chat(
        "Áo sơ mi nữ",
        conversation_id=conversation_id,
    )

    second_recommendation = (
        second["criteria"].get("size_recommendation") or {}
    )

    assert second_recommendation["recommended_size"] == "M", second
    assert second_recommendation["alternative_size"] == "L", second
    assert second["pending_action"] == "choose_fit_preference", second

    third = service.chat(
        "thích mặc rộng",
        conversation_id=conversation_id,
    )

    criteria = third["criteria"]
    recommendation = criteria.get("size_recommendation") or {}

    assert criteria["intent"] == "size_recommendation", third
    assert criteria["product_type"] == "Áo sơ mi", third
    assert criteria["gender"] == "nữ", third
    assert criteria["height_cm"] == 162, third
    assert criteria["weight_kg"] == 55, third
    assert criteria["fit_preference"] == "loose", third

    assert recommendation["recommended_size"] == "L", third
    assert recommendation["alternative_size"] == "M", third
    assert recommendation["boundary_case"] is True, third

    assert third["pending_action"] is None, third

    reply = third["reply"].lower()

    assert "size l" in reply, third["reply"]
    assert "mặc rộng" in reply, third["reply"]
    assert "bạn thích mặc ôm" not in reply, third["reply"]
    assert "kiểu dáng muốn mặc ôm/vừa/rộng" not in reply, third["reply"]


def test_color_followup_out_of_stock(service: ChatbotService) -> None:
    conversation_id = "test-color-followup"
    CONVERSATION_STORE.pop(conversation_id, None)

    service.chat(
        "Tôi muốn xem áo sơ mi Oxford",
        conversation_id=conversation_id,
    )

    result = service.chat(
        "không có màu đỏ à bạn?",
        conversation_id=conversation_id,
    )

    criteria = result["criteria"]
    availability = criteria.get("availability") or {}

    assert criteria["intent"] == "variant_availability_check"
    assert criteria["color"] == "đỏ đô"
    assert availability["status"] == "out_of_stock"
    assert result["products"] == [], result
    assert result["product_ids"] == [], result
    assert "đỏ đô" in result["reply"].lower()
    assert "hết hàng" in result["reply"].lower()


def run_phase1_product_advice_cases(service: ChatbotService) -> None:
    conversation_id = "phase1_multi_turn_preferences"
    CONVERSATION_STORE.pop(conversation_id, None)

    start = service.chat("Tôi muốn mua áo sơ mi", top_k=5, conversation_id=conversation_id)
    assert start["products"] == [], start
    assert start["product_ids"] == [], start
    assert start["pending_action"] == "collect_product_preferences", start
    assert start["criteria"].get("product_type") == "Áo sơ mi", start
    assert "ngân sách" in start["reply"].lower(), start["reply"]

    gender = service.chat("Cho nam", top_k=5, conversation_id=conversation_id)
    assert gender["products"] == [], gender
    assert gender["pending_action"] == "collect_product_preferences", gender
    assert gender["criteria"].get("gender") == "nam", gender
    assert gender["conversation_memory"].get("gender") == "nam", gender
    assert "nam hay nữ" not in gender["reply"].lower(), gender["reply"]
    assert "đi làm" in gender["reply"].lower() and "ngân sách" in gender["reply"].lower(), gender["reply"]

    use_case = service.chat("Đi làm", top_k=5, conversation_id=conversation_id)
    assert use_case["products"] == [], use_case
    assert use_case["pending_action"] == "collect_product_preferences", use_case
    assert use_case["criteria"].get("gender") == "nam", use_case
    assert use_case["criteria"].get("use_case") == "đi làm", use_case
    assert use_case["conversation_memory"].get("use_case") == "đi làm", use_case
    assert "nam hay nữ" not in use_case["reply"].lower(), use_case["reply"]
    assert "ngân sách" in use_case["reply"].lower(), use_case["reply"]

    budget = service.chat("Khoảng 300 nghìn", top_k=5, conversation_id=conversation_id)
    criteria = budget["criteria"]
    memory = budget["conversation_memory"]
    assert budget["pending_action"] is None, budget
    assert criteria.get("product_type") == "Áo sơ mi", criteria
    assert criteria.get("gender") == "nam", criteria
    assert criteria.get("use_case") == "đi làm", criteria
    assert criteria.get("budget_max_vnd") == 300000, criteria
    assert memory.get("product_type") == "Áo sơ mi", memory
    assert memory.get("gender") == "nam", memory
    assert memory.get("use_case") == "đi làm", memory
    assert memory.get("budget_max_vnd") == 300000, memory
    assert budget["product_ids"], budget
    assert all(product["effective_price_vnd"] <= 300000 for product in budget["products"]), budget
    assert "rag" not in budget["reply"].lower() and "score" not in budget["reply"].lower(), budget["reply"]
    print_result("FLOW J: PHASE 1 MULTI-TURN PREFERENCE MERGE", "Khoảng 300 nghìn", budget)

    one_shot = service.chat(
        "Tôi cần áo sơ mi nam đi làm màu trắng dưới 400 nghìn",
        top_k=5,
        conversation_id="phase1_one_shot",
    )
    one_criteria = one_shot["criteria"]
    assert one_shot["pending_action"] is None, one_shot
    assert one_criteria.get("intent") == "product_search", one_criteria
    assert one_criteria.get("product_type") == "Áo sơ mi", one_criteria
    assert one_criteria.get("gender") == "nam", one_criteria
    assert one_criteria.get("use_case") == "đi làm", one_criteria
    assert one_criteria.get("color") == "trắng", one_criteria
    assert one_criteria.get("budget_max_vnd") == 400000, one_criteria
    assert one_shot["product_ids"], one_shot
    assert all(product["product_type"] == "Áo sơ mi" for product in one_shot["products"]), one_shot
    assert all(product["effective_price_vnd"] <= 400000 for product in one_shot["products"]), one_shot
    assert "nam hay nữ" not in one_shot["reply"].lower(), one_shot["reply"]
    assert "ngân sách khoảng bao nhiêu" not in one_shot["reply"].lower(), one_shot["reply"]
    print_result("FLOW K: PHASE 1 ONE-SHOT SEARCH", one_shot["criteria"]["raw_message"], one_shot)

    unknown = service.chat("Tôi chưa biết mua gì", top_k=5, conversation_id="phase1_unknown")
    unknown_reply = unknown["reply"].lower()
    assert unknown["products"] == [], unknown
    assert unknown["product_ids"] == [], unknown
    assert unknown["pending_action"] == "choose_use_case", unknown
    assert unknown["criteria"].get("response_mode") == "ask_use_case", unknown
    assert "đi làm" in unknown_reply and "đi chơi" in unknown_reply and "mặc hằng ngày" in unknown_reply, unknown["reply"]
    assert "shop gợi ý" not in unknown_reply and "shop tìm được" not in unknown_reply, unknown["reply"]
    print_result("FLOW L: PHASE 1 UNKNOWN PRODUCT NEED", "Tôi chưa biết mua gì", unknown)

    low_budget = service.chat(
        "Mình cần áo sơ mi nam đi làm dưới 250 nghìn",
        top_k=5,
        conversation_id="phase1_low_budget",
    )
    low_reply = low_budget["reply"].lower()
    assert low_budget["products"], low_budget
    assert low_budget["product_ids"], low_budget
    assert low_budget["criteria"].get("response_mode") == "nearest_over_budget_product", low_budget
    assert low_budget["criteria"].get("nearest_over_budget_products"), low_budget
    assert "chưa có mẫu áo sơ mi đi làm nào đúng mức" in low_reply, low_budget["reply"]
    assert "mẫu gần ngân sách nhất" in low_reply, low_budget["reply"]
    assert "cao hơn ngân sách" in low_reply, low_budget["reply"]
    print_result("FLOW M: PHASE 1 NEAREST OVER BUDGET", "Mình cần áo sơ mi nam đi làm dưới 250 nghìn", low_budget)


def test_budget_fallback_nearest_over_budget_product() -> None:
    service = ChatbotService()
    products = service.data_store.products.copy()
    base = products[products["product_id"] == "P0101"].iloc[0].copy()
    base.update(
        {
            "product_id": "P0999",
            "product_name": "Váy đen dáng A",
            "brand": "AutoBiz Fashion",
            "price_vnd": 299000,
            "sale_price_vnd": 299000,
            "effective_price_vnd": 299000,
            "stock_total": 8,
            "rating": 4.5,
            "sold_30d": 20,
            "tags": "váy|đầm|nữ|đi chơi|đen",
            "short_description": "Váy đen dáng A phù hợp đi chơi",
        }
    )
    base["tags_text"] = str(base["tags"]).replace("|", " ")
    base["search_text"] = (
        "Váy đen dáng A Thời trang Váy AutoBiz Fashion "
        "váy đầm nữ đi chơi đen Váy đen dáng A phù hợp đi chơi"
    )
    products = pd.concat([products, pd.DataFrame([base])], ignore_index=True)
    service.search_engine = ProductSearchEngine(products, service.data_store.variants)

    conversation_id = "phase1_nearest_dress_budget"
    CONVERSATION_STORE.pop(conversation_id, None)
    result = service.chat("váy nữ đi chơi khoảng 200k", top_k=5, conversation_id=conversation_id)
    criteria = result["criteria"]
    nearest = criteria.get("nearest_over_budget_products") or []

    assert result["products"], result
    assert result["product_ids"] == ["P0999"], result
    assert criteria.get("response_mode") == "nearest_over_budget_product", criteria
    assert nearest and nearest[0]["product_id"] == "P0999", result
    assert nearest[0]["effective_price_vnd"] == 299000, result
    assert nearest[0]["difference_vnd"] == 99000, result
    assert result["products"][0]["effective_price_vnd"] == 299000, result
    assert result["products"][0]["difference_vnd"] == 99000, result
    reply = result["reply"].lower()
    assert "chưa có mẫu váy đi chơi nào đúng mức 200.000đ" in reply, result["reply"]
    assert "váy đen dáng a" in reply and "299.000đ" in reply, result["reply"]
    assert "99.000đ" in reply and "cao hơn ngân sách" in reply, result["reply"]
    print_result("FLOW M2: NEAREST DRESS OVER BUDGET", "váy nữ đi chơi khoảng 200k", result)


def run_phase2_cases(service: ChatbotService) -> None:
    size_context = "phase2_size_context"
    CONVERSATION_STORE.pop(size_context, None)
    start = service.chat("Tôi muốn mua áo sơ mi nam đi làm khoảng 300k", top_k=5, conversation_id=size_context)
    assert start["product_ids"] and start["product_ids"][0] == "P0018", start

    size = service.chat("Tôi cao 1m75 nặng 69kg mặc size nào?", top_k=5, conversation_id=size_context)
    assert size["criteria"].get("intent") == "size_recommendation", size
    assert size["criteria"].get("product_type") == "Áo sơ mi", size
    assert size["criteria"].get("gender") == "nam", size
    assert size["criteria"].get("height_cm") == 175, size
    assert size["criteria"].get("weight_kg") == 69, size
    assert size["criteria"].get("size_recommendation", {}).get("recommended_size") == "L", size
    assert size["pending_action"] is None, size

    no_context = "phase2_size_no_context"
    CONVERSATION_STORE.pop(no_context, None)
    no_context_size = service.chat("Tôi cao 1m75 nặng 69kg mặc size nào?", top_k=5, conversation_id=no_context)
    assert no_context_size["pending_action"] == "choose_product_for_size", no_context_size
    assert no_context_size["products"] == [], no_context_size

    boundary_context = "phase2_size_boundary"
    CONVERSATION_STORE.pop(boundary_context, None)
    service.chat("Tôi muốn mua áo sơ mi nam đi làm", top_k=5, conversation_id=boundary_context)
    boundary = service.chat("Tôi cao 1m75 nặng 75kg mặc size nào?", top_k=5, conversation_id=boundary_context)
    assert boundary["pending_action"] == "choose_fit_preference", boundary
    assert boundary["criteria"].get("size_recommendation", {}).get("boundary_case") is True, boundary
    loose = service.chat("Mình thích mặc rộng", top_k=5, conversation_id=boundary_context)
    assert loose["pending_action"] is None, loose
    assert loose["criteria"].get("fit_preference") == "loose", loose
    assert loose["criteria"].get("size_recommendation", {}).get("recommended_size") == "XL", loose

    in_stock = service.chat(
        "Áo sơ mi Oxford best seller size M màu xanh nhạt còn không?",
        top_k=5,
        conversation_id="phase2_variant_in_stock",
    )
    assert in_stock["criteria"].get("availability", {}).get("status") == "in_stock", in_stock
    assert in_stock["products"][0].get("sku") == "ABF-OXFORD-M-LIGHTBLUE", in_stock
    assert in_stock["products"][0].get("stock") == 13, in_stock

    out_stock = service.chat(
        "Áo sơ mi Oxford best seller size M màu đỏ đô còn không?",
        top_k=5,
        conversation_id="phase2_variant_out_stock",
    )
    availability = out_stock["criteria"].get("availability", {})
    assert availability.get("status") == "out_of_stock", out_stock
    assert {item.get("color") for item in availability.get("alternatives", [])} >= {"Trắng", "Xanh nhạt"}, out_stock

    reject_context = "phase2_reject"
    CONVERSATION_STORE.pop(reject_context, None)
    first = service.chat("Tôi muốn mua áo sơ mi nam đi làm khoảng 300k", top_k=5, conversation_id=reject_context)
    rejected_id = first["product_ids"][0]
    alternative = service.chat("Mẫu này không hợp, cho tôi mẫu khác", top_k=5, conversation_id=reject_context)
    assert rejected_id in alternative["conversation_memory"].get("rejected_product_ids", []), alternative
    assert rejected_id not in alternative["product_ids"], alternative
    nearest = alternative["criteria"].get("nearest_over_budget_products") or []
    if nearest:
        assert nearest[0].get("product_id") != rejected_id, alternative

    cheaper_context = "phase2_cheaper"
    CONVERSATION_STORE.pop(cheaper_context, None)
    current = service.chat("Áo thun Sunny Classic size M màu trắng còn không?", top_k=5, conversation_id=cheaper_context)
    cheaper = service.chat("Có mẫu rẻ hơn không?", top_k=5, conversation_id=cheaper_context)
    current_price = current["products"][0]["effective_price_vnd"]
    assert cheaper["criteria"].get("response_mode") == "cheaper_product", cheaper
    assert cheaper["product_ids"] and all(product["effective_price_vnd"] < current_price for product in cheaper["products"]), cheaper
    assert current["product_ids"][0] not in cheaper["product_ids"], cheaper

    color_context = "phase2_color_change"
    CONVERSATION_STORE.pop(color_context, None)
    service.chat("Áo sơ mi Oxford best seller size M màu trắng còn không?", top_k=5, conversation_id=color_context)
    color_change = service.chat("Có màu xanh nhạt không?", top_k=5, conversation_id=color_context)
    assert color_change["criteria"].get("availability", {}).get("status") == "in_stock", color_change
    assert color_change["products"][0].get("sku") == "ABF-OXFORD-M-LIGHTBLUE", color_change

    size_policy = service.chat("Có đổi size không?", top_k=5, conversation_id="phase2_policy_size")
    assert size_policy["criteria"].get("intent") == "policy_question", size_policy
    assert size_policy["criteria"].get("policy", {}).get("policy_value") is None, size_policy
    assert "chưa có thông tin" in size_policy["reply"].lower(), size_policy["reply"]

    no_policy = service.chat("Giao hàng mất bao lâu?", top_k=5, conversation_id="phase2_policy_missing")
    assert no_policy["criteria"].get("intent") == "policy_question", no_policy
    assert no_policy["criteria"].get("policy", {}).get("policy_value") is None, no_policy

    negative_context = "phase2_negative"
    CONVERSATION_STORE.pop(negative_context, None)
    picked = service.chat("Áo thun Sunny Classic size M màu trắng còn không?", top_k=5, conversation_id=negative_context)
    negative = service.chat("Đắt quá", top_k=5, conversation_id=negative_context)
    assert negative["criteria"].get("intent") == "negative_feedback", negative
    assert negative["products"] == [], negative
    assert negative["conversation_memory"].get("selected_product_id") == picked["product_ids"][0], negative


def run_phase3_order_cases(service: ChatbotService) -> None:
    original_sender = service_module.submit_draft_order_to_n8n
    webhook_calls: list[dict] = []

    def fake_success(draft_order: dict) -> SubmissionResult:
        payload = build_n8n_payload(
            draft_order,
            conversation_id=str(draft_order.get("customer_chat_id") or ""),
        )
        webhook_calls.append(payload)
        return SubmissionResult(ok=True, status_code=200, payload=payload)

    service_module.submit_draft_order_to_n8n = fake_success
    try:
        no_context_id = "phase3_no_context_purchase"
        CONVERSATION_STORE.pop(no_context_id, None)
        no_context = service.chat("Tôi muốn mua 1 cái", top_k=5, conversation_id=no_context_id)
        assert no_context["criteria"].get("purchase_intent") is True, no_context
        assert no_context["pending_action"] == "choose_order_product", no_context
        assert no_context.get("order_draft") is None, no_context
        assert "bạn muốn mua mẫu sản phẩm nào" in no_context["reply"].lower(), no_context["reply"]

        flow_id = "phase3_happy_path"
        CONVERSATION_STORE.pop(flow_id, None)
        start = service.chat(
            "Tôi muốn mua áo sơ mi nam đi làm khoảng 350 nghìn",
            top_k=5,
            conversation_id=flow_id,
        )
        assert start["product_ids"] and start["product_ids"][0] == "P0018", start

        choose = service.chat("Tôi lấy mẫu đầu tiên", top_k=5, conversation_id=flow_id)
        draft = choose["order_draft"]
        item = draft["items"][0]
        assert choose["pending_action"] == "choose_order_color_and_size", choose
        assert item["product_id"] == "P0018", choose
        assert item["sku"] is None, choose
        draft_id = draft["draft_id"]

        variant = service.chat("Trắng size M", top_k=5, conversation_id=flow_id)
        item = variant["order_draft"]["items"][0]
        assert variant["pending_action"] == "choose_order_quantity", variant
        assert item["sku"] == "ABF-OXFORD-M-WHITE", variant
        assert item["variant_stock"] == 9, variant

        too_many = service.chat("10 cái", top_k=5, conversation_id=flow_id)
        assert too_many["pending_action"] == "choose_order_quantity", too_many
        assert too_many["criteria"].get("response_mode") == "order_quantity_exceeds_stock", too_many
        assert too_many["order_draft"]["items"][0]["quantity"] != 10, too_many

        quantity = service.chat("1 cái", top_k=5, conversation_id=flow_id)
        assert quantity["pending_action"] == "collect_customer_name", quantity

        customer = service.chat(
            "Nguyễn Văn A, 0912345678, 123 Nguyễn Trãi, Hà Nội",
            top_k=5,
            conversation_id=flow_id,
        )
        customer_info = customer["order_draft"]["customer"]
        assert customer["pending_action"] == "choose_payment_method", customer
        assert customer_info["name"] == "Nguyễn Văn A", customer
        assert customer_info["phone"] == "0912345678", customer
        assert customer_info["address"] == "123 Nguyễn Trãi, Hà Nội", customer

        summary = service.chat("COD", top_k=5, conversation_id=flow_id)
        assert summary["pending_action"] == "confirm_order_draft", summary
        assert summary["order_draft"]["status"] == "awaiting_customer_confirmation", summary
        assert summary["order_draft"]["draft_id"] == draft_id, summary
        assert "đơn nháp" in summary["reply"].lower(), summary["reply"]
        assert "đặt thành công" not in summary["reply"].lower(), summary["reply"]

        edited = service.chat(
            "Đổi địa chỉ sang 50 Trần Duy Hưng, Hà Nội",
            top_k=5,
            conversation_id=flow_id,
        )
        assert edited["order_draft"]["draft_id"] == draft_id, edited
        assert edited["order_draft"]["customer"]["address"] == "50 Trần Duy Hưng, Hà Nội", edited
        assert edited["pending_action"] == "confirm_order_draft", edited

        edited_variant = service.chat(
            "Đổi màu sang xanh nhạt size M",
            top_k=5,
            conversation_id=flow_id,
        )
        assert edited_variant["order_draft"]["draft_id"] == draft_id, edited_variant
        assert edited_variant["order_draft"]["items"][0]["sku"] == "ABF-OXFORD-M-LIGHTBLUE", edited_variant
        assert edited_variant["order_draft"]["items"][0]["variant_stock"] == 13, edited_variant

        submitted = service.chat("Xác nhận", top_k=5, conversation_id=flow_id)
        assert submitted["pending_action"] is None, submitted
        assert submitted["order_draft"]["status"] == "pending_shop_approval", submitted
        assert "gửi đơn nháp sang shop" in submitted["reply"].lower(), submitted["reply"]
        assert "đặt thành công" not in submitted["reply"].lower(), submitted["reply"]
        assert len(webhook_calls) == 1, webhook_calls
        payload = webhook_calls[0]
        assert payload["event"] == "draft_order_confirmed", payload
        assert payload["draft_order_id"] == draft_id, payload
        assert payload["draft_id"] == draft_id, payload
        assert payload["status"] == "pending_owner_review", payload
        assert payload["items"][0]["sku"] == "ABF-OXFORD-M-LIGHTBLUE", payload
        assert payload["items"][0]["stock_at_customer_confirmation"] == 13, payload
        assert payload["payment_method"] == "cod", payload
        forbidden_payload_keys = {"rag_contexts", "debug", "score", "embedding", "internal_scores"}
        assert forbidden_payload_keys.isdisjoint(payload.keys()), payload

        invalid_phone_id = "phase3_invalid_phone"
        CONVERSATION_STORE.pop(invalid_phone_id, None)
        service.chat(
            "Áo sơ mi Oxford best seller size M màu trắng còn không?",
            top_k=5,
            conversation_id=invalid_phone_id,
        )
        service.chat("Tôi lấy mẫu này", top_k=5, conversation_id=invalid_phone_id)
        service.chat("1 cái", top_k=5, conversation_id=invalid_phone_id)
        service.chat("Nguyễn Văn A", top_k=5, conversation_id=invalid_phone_id)
        invalid_phone = service.chat("12345", top_k=5, conversation_id=invalid_phone_id)
        assert invalid_phone["pending_action"] == "collect_customer_phone", invalid_phone
        assert invalid_phone["criteria"].get("response_mode") == "invalid_customer_phone", invalid_phone

        out_stock_id = "phase3_out_stock"
        CONVERSATION_STORE.pop(out_stock_id, None)
        service.chat(
            "Áo sơ mi Oxford best seller size M màu đỏ đô còn không?",
            top_k=5,
            conversation_id=out_stock_id,
        )
        out_stock = service.chat("Tôi lấy mẫu này", top_k=5, conversation_id=out_stock_id)
        assert out_stock["pending_action"] == "choose_order_color_and_size", out_stock
        assert out_stock["criteria"].get("response_mode") == "order_variant_out_of_stock", out_stock
        assert "hết hàng" in out_stock["reply"].lower(), out_stock["reply"]
        assert (out_stock["order_draft"]["customer"] or {}).get("name") is None, out_stock

        cancel_id = "phase3_cancel"
        CONVERSATION_STORE.pop(cancel_id, None)
        service.chat(
            "Áo sơ mi Oxford best seller size M màu trắng còn không?",
            top_k=5,
            conversation_id=cancel_id,
        )
        service.chat("Tôi lấy mẫu này", top_k=5, conversation_id=cancel_id)
        cancelled = service.chat("Thôi không mua nữa", top_k=5, conversation_id=cancel_id)
        assert cancelled["pending_action"] is None, cancelled
        assert cancelled["order_draft"]["status"] == "cancelled", cancelled

        policy_id = "phase3_policy_during_order"
        CONVERSATION_STORE.pop(policy_id, None)
        service.chat(
            "Áo sơ mi Oxford best seller size M màu trắng còn không?",
            top_k=5,
            conversation_id=policy_id,
        )
        service.chat("Tôi lấy mẫu này", top_k=5, conversation_id=policy_id)
        service.chat("1 cái", top_k=5, conversation_id=policy_id)
        policy = service.chat("Shop có đổi size không?", top_k=5, conversation_id=policy_id)
        assert policy["pending_action"] == "collect_customer_name", policy
        assert policy["criteria"].get("response_mode") == "policy_question_during_order", policy

        browse_shift = service.chat("Cho tôi xem mẫu khác", top_k=5, conversation_id=policy_id)
        assert browse_shift["pending_action"] == "collect_customer_name", browse_shift
        assert "hủy đơn nháp hiện tại" in browse_shift["reply"].lower(), browse_shift["reply"]

        retry_calls: list[dict] = []

        def fake_retry(draft_order: dict) -> SubmissionResult:
            payload = build_n8n_payload(
                draft_order,
                conversation_id=str(draft_order.get("customer_chat_id") or ""),
            )
            retry_calls.append(payload)
            if len(retry_calls) == 1:
                return SubmissionResult(ok=False, error_type="timeout", payload=payload)
            return SubmissionResult(ok=True, status_code=200, payload=payload)

        service_module.submit_draft_order_to_n8n = fake_retry
        retry_id = "phase3_webhook_retry"
        CONVERSATION_STORE.pop(retry_id, None)
        for message in [
            "Áo sơ mi Oxford best seller size M màu trắng còn không?",
            "Tôi lấy mẫu này",
            "1 cái",
            "Nguyễn Văn A, 0912345678, 123 Nguyễn Trãi, Hà Nội",
            "COD",
        ]:
            service.chat(message, top_k=5, conversation_id=retry_id)
        failed = service.chat("Xác nhận", top_k=5, conversation_id=retry_id)
        failed_draft_id = failed["order_draft"]["draft_id"]
        assert failed["pending_action"] == "confirm_order_draft", failed
        assert failed["order_draft"]["status"] == "submission_failed", failed
        assert failed["criteria"].get("response_mode") == "order_draft_submission_failed", failed
        retried = service.chat("Gửi lại", top_k=5, conversation_id=retry_id)
        assert retried["pending_action"] is None, retried
        assert retried["order_draft"]["draft_id"] == failed_draft_id, retried
        assert retried["order_draft"]["status"] == "pending_shop_approval", retried
        assert len(retry_calls) == 2, retry_calls
    finally:
        service_module.submit_draft_order_to_n8n = original_sender


def run_catalog_overview_cases(service: ChatbotService) -> None:
    for index, message in enumerate(CATALOG_OVERVIEW_QUESTIONS, start=1):
        result = service.chat(message, top_k=5, conversation_id=f"overview_{index}")
        validate_result(message, result)


def run_rag_checks(service: ChatbotService) -> None:
    stats = service.rebuild_rag_index(force_rebuild=True)
    assert stats.get("total_documents", 0) > 0, stats
    assert stats.get("product_docs", 0) > 0, stats
    assert stats.get("variant_docs", 0) > 0, stats
    assert stats.get("policy_docs", 0) > 0, stats
    assert stats.get("size_chart_docs", 0) > 0, stats
    assert stats.get("catalog_scope_docs", 0) == 1, stats

    retriever = AutoBizRAGRetriever()
    product_query = "Áo thun basic unisex màu mới size M sọc xanh trắng"
    product_results = retriever.retrieve(product_query, top_k=5)
    assert product_results, product_results
    assert any(
        (result.get("metadata") or {}).get("product_id") == "P0002"
        or (result.get("metadata") or {}).get("sku") == "ABF-0002-M-613"
        for result in product_results
    ), product_results

    scope_query = "bánh mì nước cam"
    scope_results = retriever.retrieve(scope_query, top_k=5)
    assert any(
        (result.get("metadata") or {}).get("doc_type") == "catalog_scope"
        for result in scope_results
    ), scope_results
    assert not any(
        (result.get("metadata") or {}).get("category_code") == "Food"
        for result in scope_results
    ), scope_results

    available_message = "Shop còn Áo thun basic unisex màu mới size M màu sọc xanh trắng không?"
    available_result = service.chat(available_message, top_k=5, conversation_id="rag_available")
    assert available_result["products"][0]["sku"] == "ABF-0002-M-613", available_result
    assert available_result["products"][0]["stock"] == 34, available_result
    assert any(
        context.get("product_id") == "P0002" or context.get("sku") == "ABF-0002-M-613"
        for context in available_result.get("rag_contexts", [])
    ), available_result.get("rag_contexts")

    out_of_scope_message = "Shop có bán bánh mì và nước cam không?"
    out_of_scope_result = service.chat(out_of_scope_message, top_k=5, conversation_id="rag_scope")
    assert out_of_scope_result["products"] == [], out_of_scope_result
    assert any(
        context.get("doc_type") == "catalog_scope"
        for context in out_of_scope_result.get("rag_contexts", [])
    ), out_of_scope_result.get("rag_contexts")

    print("=" * 100)
    print("RAG STATS")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    print("RAG SEARCH SAMPLE")
    print(json.dumps(product_results[:3], ensure_ascii=False, indent=2))


def main() -> None:
    service = ChatbotService()
    run_rag_checks(service)
    for question in QUESTIONS:
        result = service.chat(question, top_k=5)
        validate_result(question, result)
        print_result("SINGLE TURN CASE", question, result)
    run_conversation_flow(service)
    run_choose_product_type_flow(service)
    run_invalid_product_type_flow(service)
    run_general_buying_and_size_cases(service)
    run_size_recommender_boundary_cases(service)
    test_recommend_size_loose_prefers_larger_boundary_size()
    test_size_followup_loose_preference(service)
    test_color_followup_out_of_stock(service)
    run_phase1_product_advice_cases(service)
    test_budget_fallback_nearest_over_budget_product()
    run_phase2_cases(service)
    run_phase3_order_cases(service)
    run_catalog_overview_cases(service)
    print("=" * 100)
    print("CHECKS OK")


if __name__ == "__main__":
    main()
