from __future__ import annotations

import json
import os
import socket
import time
import uuid
from typing import Any

import gradio as gr

from app.service import ChatbotService
from app.order_draft_service import masked_order_draft

bot = ChatbotService()

DEFAULT_SERVER_PORTS = [7860, 7861, 7862, 7863, 7864, 7865]
DEV_MODE_TRUE_VALUES = {"true", "1", "yes"}
GRADIO_UI_EVENT_OPTIONS = {"api_name": False, "show_api": False}


def is_dev_mode() -> bool:
    ui_mode = os.getenv("AUTOBIZ_UI_MODE", "").strip().lower()
    if ui_mode:
        return ui_mode == "dev"
    return os.getenv("AUTOBIZ_DEV_MODE", "").strip().lower() in DEV_MODE_TRUE_VALUES


DEV_MODE = is_dev_mode()

SUGGESTED_MESSAGES = {
    "Test còn hàng P0002": "Shop còn Áo thun basic unisex màu mới size M màu sọc xanh trắng không?",
    "Test hết hàng Oxford đỏ đô": "Áo sơ mi Oxford best seller size M màu đỏ đô còn không?",
    "Test mixed request": (
        "Tôi muốn mua Áo sơ mi denim mỏng mùa hè size L màu xanh navy "
        "và thêm giày sneaker trắng, shop có không?"
    ),
    "Test ngoài phạm vi": "Shop có bán bánh mì và nước cam không?",
    "Test áo ba lỗ nam": "Tôi cần mua một cái áo ba lỗ dành cho nam màu trắng để mặc thoải mái ở nhà",
    "Test quần áo đá bóng": "Shop mình có bán mấy sản phẩm quần áo đá bóng không",
    "Test follow-up Có": "Có",
    "Test chọn Áo thun": "Áo thun",
    "Test nam trắng dưới 300k": "nam màu trắng dưới 300k",
    "Test váy đẹp": "Có váy nào đẹp không?",
}

USER_SUGGESTED_MESSAGES = {
    "Gợi ý đồ đi chơi": "gợi ý đồ đi chơi",
    "Áo thun nam trắng dưới 300k": "áo thun nam màu trắng dưới 300k",
    "Shop có bán quần áo đá bóng không?": "Shop có bán quần áo đá bóng không?",
    "Áo sơ mi Oxford size M màu đỏ đô còn không?": "Áo sơ mi Oxford best seller size M màu đỏ đô còn không?",
    "Váy đi chơi": "váy đi chơi",
    "Set bộ mặc hằng ngày": "set bộ mặc hằng ngày",
}

ACTIVE_SUGGESTED_MESSAGES = SUGGESTED_MESSAGES if DEV_MODE else USER_SUGGESTED_MESSAGES
SUGGESTED_TITLE = "### Test nhanh" if DEV_MODE else "### Gợi ý nhanh"


def new_conversation_id() -> str:
    return f"gradio_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def money(vnd: int | float | None) -> str:
    if vnd is None:
        return "chưa rõ giá"
    return f"{int(vnd):,}".replace(",", ".") + "đ"


def format_bot_message(response: dict[str, Any]) -> str:
    return response.get("reply", "")


def format_debug(response: dict[str, Any]) -> str:
    criteria = response.get("criteria") or {}
    memory = response.get("conversation_memory") or bot.conversation_memory(
        str(response.get("conversation_id") or "")
    )
    safe_criteria = dict(criteria)
    if safe_criteria.get("order_draft"):
        safe_criteria["order_draft"] = masked_order_draft(safe_criteria.get("order_draft"))
    safe_memory = dict(memory)
    if safe_memory.get("order_draft"):
        safe_memory["order_draft"] = masked_order_draft(safe_memory.get("order_draft"))
    debug_payload = {
        "conversation_id": response.get("conversation_id"),
        "pending_action": response.get("pending_action"),
        "intent": criteria.get("intent"),
        "response_mode": criteria.get("response_mode"),
        "last_product_type": memory.get("last_product_type") or criteria.get("product_type"),
        "last_product_ids": memory.get("last_product_ids", []),
        "height_cm": criteria.get("height_cm") or memory.get("height_cm"),
        "weight_kg": criteria.get("weight_kg") or memory.get("weight_kg"),
        "gender": criteria.get("gender") or memory.get("gender"),
        "use_case": criteria.get("use_case") or memory.get("use_case"),
        "criteria": safe_criteria,
        "product_ids": response.get("product_ids", []),
        "rag_contexts": response.get("rag_contexts", []),
        "rag_status": criteria.get("rag_status"),
        "order_draft": masked_order_draft(response.get("order_draft")),
    }
    return json.dumps(debug_payload, ensure_ascii=False, indent=2)


def format_debug_for_current_mode(response: dict[str, Any]) -> str:
    if not DEV_MODE:
        return "{}"
    return format_debug(response)


def format_memory_for_current_mode(response: dict[str, Any]) -> str:
    if not DEV_MODE:
        return "{}"
    memory = response.get("conversation_memory") or bot.conversation_memory(
        str(response.get("conversation_id") or "")
    )
    memory = dict(memory)
    if memory.get("order_draft"):
        memory["order_draft"] = masked_order_draft(memory.get("order_draft"))
    return json.dumps(memory, ensure_ascii=False, indent=2)


def format_criteria_for_current_mode(response: dict[str, Any]) -> str:
    if not DEV_MODE:
        return "{}"
    criteria = dict(response.get("criteria") or {})
    if criteria.get("order_draft"):
        criteria["order_draft"] = masked_order_draft(criteria.get("order_draft"))
    return json.dumps(criteria, ensure_ascii=False, indent=2)


def format_order_draft_for_current_mode(response: dict[str, Any]) -> str:
    if not DEV_MODE:
        return "{}"
    draft = response.get("order_draft")
    if draft is None:
        memory = response.get("conversation_memory") or bot.conversation_memory(
            str(response.get("conversation_id") or "")
        )
        draft = memory.get("order_draft")
    return json.dumps(masked_order_draft(draft) or {}, ensure_ascii=False, indent=2)


def format_product_ids_for_current_mode(response: dict[str, Any]) -> str:
    if not DEV_MODE:
        return "[]"
    return json.dumps(response.get("product_ids") or [], ensure_ascii=False, indent=2)


def format_rag_contexts_for_current_mode(response: dict[str, Any]) -> str:
    if not DEV_MODE:
        return "[]"
    return json.dumps(response.get("rag_contexts") or [], ensure_ascii=False, indent=2)


def compact_rag_status(status: dict[str, Any] | None = None) -> dict[str, Any]:
    status = status or bot.rag_status()
    return {
        "ready": status.get("ready"),
        "backend": status.get("backend"),
        "embedding_backend": status.get("embedding_backend"),
        "total_documents": status.get("total_documents", 0),
        "embedding_error": status.get("embedding_error"),
    }


def format_rag_status_json_for_current_mode(response: dict[str, Any] | None = None) -> str:
    if not DEV_MODE:
        return "{}"
    criteria = (response or {}).get("criteria") or {}
    status = criteria.get("rag_status") or bot.rag_status()
    return json.dumps(compact_rag_status(status), ensure_ascii=False, indent=2)


def format_rag_status(status: dict[str, Any] | None = None) -> str:
    status = status or bot.rag_status()
    ready_text = "RAG index ready" if status.get("ready") else "RAG index not ready"
    backend = status.get("backend") or "unknown"
    embedding_backend = status.get("embedding_backend") or "unknown"
    total_documents = status.get("total_documents", 0)
    return f"{ready_text} | docs: {total_documents} | backend: {backend} | embeddings: {embedding_backend}"


def format_rebuild_status(status: dict[str, Any]) -> str:
    backend = status.get("backend") or "unknown"
    embedding_backend = status.get("embedding_backend") or "unknown"
    total_documents = status.get("total_documents", 0)
    return f"RAG index rebuilt successfully | docs: {total_documents} | backend: {backend} | embeddings: {embedding_backend}"


def rebuild_rag_index() -> tuple[str, str, str]:
    stats = bot.rebuild_rag_index(force_rebuild=True)
    return (
        format_rebuild_status(stats),
        json.dumps(stats, ensure_ascii=False, indent=2),
        json.dumps(compact_rag_status(stats), ensure_ascii=False, indent=2),
    )


def refresh_rag_status() -> tuple[str, str]:
    status = bot.rag_status()
    status_json = json.dumps(compact_rag_status(status), ensure_ascii=False, indent=2)
    return status_json, status_json


def is_port_available(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def select_server_port() -> int:
    configured_port = os.getenv("GRADIO_SERVER_PORT")
    candidate_ports = DEFAULT_SERVER_PORTS
    if configured_port:
        env_port = int(configured_port)
        candidate_ports = [env_port, *[port for port in DEFAULT_SERVER_PORTS if port != env_port]]

    for port in candidate_ports:
        if is_port_available(port):
            return port
    raise RuntimeError(f"No available Gradio server port found in {candidate_ports}")


def send_message(
    user_message: str,
    history: list[dict[str, str]] | None,
    conversation_id: str | None,
) -> tuple[str, list[dict[str, str]], str, str, str, str, str, str, str, str, str]:
    conversation_id = conversation_id or new_conversation_id()
    history = history or []
    user_message = (user_message or "").strip()

    if not user_message:
        return "", history, "{}", "{}", "{}", "{}", "[]", "[]", "{}", conversation_id, conversation_id

    response = bot.chat(
        message=user_message,
        top_k=5,
        conversation_id=conversation_id,
    )
    updated_history = [
        *history,
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": format_bot_message(response)},
    ]
    return (
        "",
        updated_history,
        format_debug_for_current_mode(response),
        format_memory_for_current_mode(response),
        format_criteria_for_current_mode(response),
        format_order_draft_for_current_mode(response),
        format_product_ids_for_current_mode(response),
        format_rag_contexts_for_current_mode(response),
        format_rag_status_json_for_current_mode(response),
        conversation_id,
        conversation_id,
    )


def make_suggested_sender(prompt: str):
    def _send(
        history: list[dict[str, str]] | None,
        conversation_id: str | None,
    ) -> tuple[str, list[dict[str, str]], str, str, str, str, str, str, str, str, str]:
        return send_message(prompt, history, conversation_id)

    return _send


def reset_chat() -> tuple[str, list[dict[str, str]], str, str, str, str, str, str, str, str, str]:
    conversation_id = new_conversation_id()
    return "", [], "{}", "{}", "{}", "{}", "[]", "[]", "{}", conversation_id, conversation_id


def init_chat() -> tuple[str, str, str, str, str, str, str, str, str, list[dict[str, str]]]:
    conversation_id = new_conversation_id()
    return conversation_id, conversation_id, "{}", "{}", "{}", "{}", "[]", "[]", "{}", []


with gr.Blocks(title="AutoBiz Product Chatbot") as demo:
    gr.Markdown("# AutoBiz Product Chatbot")
    gr.Markdown("Tư vấn sản phẩm theo nhu cầu khách hàng")

    conversation_state = gr.State()

    with gr.Row():
        conversation_box = gr.Textbox(
            label="conversation_id",
            value="",
            interactive=False,
            visible=DEV_MODE,
            scale=2,
        )

    chatbot = gr.Chatbot(
        label="Hội thoại",
        type="messages",
        height=480,
    )

    with gr.Row():
        message_box = gr.Textbox(
            label="Tin nhắn",
            placeholder="Nhập nhu cầu của khách hàng...",
            lines=2,
            scale=5,
        )
        send_button = gr.Button("Gửi", variant="primary", scale=1)
        new_chat_button = gr.Button("New chat", scale=1)

    debug_panel = gr.Code(
        label="Debug panel",
        language="json",
        value="{}",
        visible=DEV_MODE,
    )
    memory_panel = gr.Code(
        label="Conversation Memory",
        language="json",
        value="{}",
        visible=DEV_MODE,
    )
    criteria_panel = gr.Code(
        label="Criteria",
        language="json",
        value="{}",
        visible=DEV_MODE,
    )
    order_draft_panel = gr.Code(
        label="Order Draft",
        language="json",
        value="{}",
        visible=DEV_MODE,
    )
    product_ids_panel = gr.Code(
        label="Product IDs",
        language="json",
        value="[]",
        visible=DEV_MODE,
    )
    rag_contexts_panel = gr.Code(
        label="RAG Contexts",
        language="json",
        value="[]",
        visible=DEV_MODE,
    )
    rag_status_json_panel = gr.Code(
        label="RAG Status",
        language="json",
        value="{}",
        visible=DEV_MODE,
    )

    if DEV_MODE:
        with gr.Row():
            rag_status_box = gr.Textbox(
                label="RAG status",
                value=format_rag_status(),
                interactive=False,
                scale=4,
            )
            rebuild_rag_button = gr.Button("Rebuild RAG Index", scale=1)
            refresh_rag_button = gr.Button("Refresh RAG Status", scale=1)

    gr.Markdown(SUGGESTED_TITLE)
    with gr.Row():
        for label, prompt in ACTIVE_SUGGESTED_MESSAGES.items():
            button = gr.Button(label)
            button.click(
                fn=make_suggested_sender(prompt),
                inputs=[chatbot, conversation_state],
                outputs=[
                    message_box,
                    chatbot,
                    debug_panel,
                    memory_panel,
                    criteria_panel,
                    order_draft_panel,
                    product_ids_panel,
                    rag_contexts_panel,
                    rag_status_json_panel,
                    conversation_state,
                    conversation_box,
                ],
                **GRADIO_UI_EVENT_OPTIONS,
            )

    send_button.click(
        fn=send_message,
        inputs=[message_box, chatbot, conversation_state],
        outputs=[
            message_box,
            chatbot,
            debug_panel,
            memory_panel,
            criteria_panel,
            order_draft_panel,
            product_ids_panel,
            rag_contexts_panel,
            rag_status_json_panel,
            conversation_state,
            conversation_box,
        ],
        **GRADIO_UI_EVENT_OPTIONS,
    )
    message_box.submit(
        fn=send_message,
        inputs=[message_box, chatbot, conversation_state],
        outputs=[
            message_box,
            chatbot,
            debug_panel,
            memory_panel,
            criteria_panel,
            order_draft_panel,
            product_ids_panel,
            rag_contexts_panel,
            rag_status_json_panel,
            conversation_state,
            conversation_box,
        ],
        **GRADIO_UI_EVENT_OPTIONS,
    )
    new_chat_button.click(
        fn=reset_chat,
        inputs=[],
        outputs=[
            message_box,
            chatbot,
            debug_panel,
            memory_panel,
            criteria_panel,
            order_draft_panel,
            product_ids_panel,
            rag_contexts_panel,
            rag_status_json_panel,
            conversation_state,
            conversation_box,
        ],
        **GRADIO_UI_EVENT_OPTIONS,
    )
    if DEV_MODE:
        rebuild_rag_button.click(
            fn=rebuild_rag_index,
            inputs=[],
            outputs=[rag_status_box, debug_panel, rag_status_json_panel],
            **GRADIO_UI_EVENT_OPTIONS,
        )
        refresh_rag_button.click(
            fn=refresh_rag_status,
            inputs=[],
            outputs=[rag_status_box, rag_status_json_panel],
            **GRADIO_UI_EVENT_OPTIONS,
        )
    demo.load(
        fn=init_chat,
        inputs=[],
        outputs=[
            conversation_state,
            conversation_box,
            debug_panel,
            memory_panel,
            criteria_panel,
            order_draft_panel,
            product_ids_panel,
            rag_contexts_panel,
            rag_status_json_panel,
            chatbot,
        ],
        **GRADIO_UI_EVENT_OPTIONS,
    )


# Run user mode (PowerShell):
# $env:AUTOBIZ_UI_MODE="user"
# py -m app.gradio_ui
#
# Run dev mode (PowerShell):
# $env:AUTOBIZ_UI_MODE="dev"
# py -m app.gradio_ui
if __name__ == "__main__":
    selected_port = select_server_port()
    demo.launch(server_name="127.0.0.1", server_port=selected_port, share=True)
