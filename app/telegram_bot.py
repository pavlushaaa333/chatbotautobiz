from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

import requests
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from app.service import ChatbotService

# =========================================================
# LOAD ENVIRONMENT VARIABLES
# =========================================================

load_dotenv()

BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
N8N_ORDER_WEBHOOK_URL = os.getenv("N8N_ORDER_WEBHOOK_URL", "").strip()
N8N_ORDER_WEBHOOK_SECRET = os.getenv("N8N_ORDER_WEBHOOK_SECRET", "").strip()


# =========================================================
# LOGGING
# =========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)


# =========================================================
# INITIALIZE AUTOBIZ
# =========================================================

# Khởi tạo một lần khi chương trình bắt đầu.
# Không khởi tạo lại ChatbotService cho từng tin nhắn.
bot = ChatbotService()


# =========================================================
# TELEGRAM UTILITIES
# =========================================================


def build_conversation_id(update: Update) -> str:
    """
    Tạo conversation_id ổn định theo Telegram chat.

    Nhờ sử dụng cùng một conversation_id, AutoBiz có thể giữ
    ngữ cảnh giữa các tin nhắn tiếp nối của cùng một khách.
    """
    if update.effective_chat is None:
        raise ValueError("Không xác định được Telegram chat")

    return f"telegram_{update.effective_chat.id}"


def extract_reply(response: dict[str, Any]) -> str:
    """
    Lấy nội dung phản hồi dành cho khách từ kết quả ChatbotService.

    Gradio hiện tại cũng sử dụng trường response['reply'].
    """
    reply = response.get("reply")

    if reply is None:
        return ""

    return str(reply).strip()


async def send_long_message(
    update: Update,
    text: str,
    max_length: int = 4000,
) -> None:
    """
    Telegram giới hạn độ dài mỗi tin nhắn.
    Hàm này tự chia phản hồi dài thành nhiều tin nhắn.
    """
    if update.message is None:
        return

    text = text.strip()

    if not text:
        return

    for start in range(0, len(text), max_length):
        chunk = text[start : start + max_length]
        await update.message.reply_text(chunk)


# =========================================================
# AUTOBIZ CHATBOT
# =========================================================


def call_autobiz(
    user_message: str,
    conversation_id: str,
) -> dict[str, Any]:
    """
    Gọi đúng pipeline mà giao diện Gradio đang sử dụng.
    """
    response = bot.chat(
        message=user_message,
        top_k=5,
        conversation_id=conversation_id,
    )

    if not isinstance(response, dict):
        raise TypeError(
            "ChatbotService.chat() phải trả về dict, "
            f"nhưng nhận được {type(response).__name__}"
        )

    return response


# =========================================================
# N8N INTEGRATION
# =========================================================


def should_send_to_n8n(response: dict[str, Any]) -> bool:
    """
    Xác định khi nào kết quả AutoBiz cần được gửi sang n8n.

    Hiện tại ưu tiên các trường hợp:
    - AutoBiz đã tạo order_draft.
    - pending_action thông báo cần tạo hoặc gửi đơn nháp.

    Có thể bổ sung pending_action khác sau khi kiểm tra
    response thực tế của luồng chốt đơn.
    """
    return False


def build_n8n_payload(
    update: Update,
    response: dict[str, Any],
    conversation_id: str,
    user_message: str,
) -> dict[str, Any]:
    """
    Chuẩn hóa dữ liệu AutoBiz gửi sang webhook của n8n.
    """
    telegram_chat_id = (
        update.effective_chat.id if update.effective_chat is not None else None
    )

    telegram_user_id = (
        update.effective_user.id if update.effective_user is not None else None
    )

    telegram_username = (
        update.effective_user.username if update.effective_user is not None else None
    )

    telegram_full_name = (
        update.effective_user.full_name if update.effective_user is not None else None
    )

    return {
        "event": "create_draft_order",
        "source": "telegram",
        "conversation_id": conversation_id,
        "telegram": {
            "chat_id": telegram_chat_id,
            "user_id": telegram_user_id,
            "username": telegram_username,
            "full_name": telegram_full_name,
        },
        "user_message": user_message,
        "reply": response.get("reply"),
        "pending_action": response.get("pending_action"),
        "order_draft": response.get("order_draft"),
        "product_ids": response.get("product_ids", []),
        "criteria": response.get("criteria", {}),
        "conversation_memory": response.get(
            "conversation_memory",
            {},
        ),
    }


def send_to_n8n(payload: dict[str, Any]) -> None:
    """
    Gửi sự kiện từ AutoBiz sang webhook n8n.
    """
    if not N8N_ORDER_WEBHOOK_URL:
        logger.warning(
            "Chưa cấu hình N8N_ORDER_WEBHOOK_URL. Bỏ qua việc gửi dữ liệu sang n8n."
        )
        return

    headers = {
        "Content-Type": "application/json",
    }

    if N8N_ORDER_WEBHOOK_SECRET:
        headers["x-autobiz-secret"] = N8N_ORDER_WEBHOOK_SECRET

    webhook_response = requests.post(
        N8N_ORDER_WEBHOOK_URL,
        json=payload,
        headers=headers,
        timeout=10,
    )

    logger.info(
        "Đã gửi sự kiện sang n8n. Status code: %s, response: %s",
        webhook_response.status_code,
        webhook_response.text,
    )

    webhook_response.raise_for_status()


# =========================================================
# TELEGRAM COMMAND HANDLERS
# =========================================================


async def new_chat_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Hiện tại conversation_id được gắn với chat_id.

    Lệnh này chủ yếu thông báo cho người dùng bắt đầu yêu cầu mới.
    Nếu ChatbotService có hàm reset memory riêng, có thể gọi thêm
    hàm reset tại đây sau.
    """
    if update.message is None:
        return

    await update.message.reply_text(
        "Được rồi, bạn hãy gửi nhu cầu mới để AutoBiz tư vấn."
    )


async def reset_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Xóa phiên tư vấn và đơn nháp của Telegram chat hiện tại.
    """
    if update.message is None:
        return

    try:
        conversation_id = build_conversation_id(update)
        await asyncio.to_thread(
            bot.reset_conversation,
            conversation_id,
        )
        await update.message.reply_text(
            "Đã xóa phiên tư vấn và đơn nháp hiện tại.\n\n"
            "Bạn muốn tìm sản phẩm gì ạ?"
        )
    except Exception:
        logger.exception("Không thể reset phiên chat khi nhận /reset.")
        await update.message.reply_text(
            "Mình chưa thể làm mới cuộc trò chuyện lúc này. Bạn thử lại sau giúp mình nhé."
        )


async def start_command(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Xử lý lệnh /start và xóa phiên tư vấn hiện tại trước khi chào.
    """
    if update.message is None:
        return

    try:
        conversation_id = build_conversation_id(update)
        await asyncio.to_thread(
            bot.reset_conversation,
            conversation_id,
        )
    except Exception:
        logger.exception("Không thể reset phiên chat khi nhận /start.")
        await update.message.reply_text(
            "Mình chưa thể làm mới cuộc trò chuyện lúc này. Bạn thử lại sau giúp mình nhé."
        )
        return

    await update.message.reply_text(
        "Xin chào! Tôi là trợ lý tư vấn AutoBiz.\n" "Bạn đang muốn tìm sản phẩm gì?"
    )


# =========================================================
# TELEGRAM MESSAGE HANDLER
# =========================================================


async def handle_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Nhận tin nhắn Telegram, gọi AutoBiz và trả câu trả lời cho khách.
    """
    if (
        update.message is None
        or update.message.text is None
        or update.effective_chat is None
    ):
        return

    user_message = update.message.text.strip()

    if not user_message:
        return

    conversation_id = build_conversation_id(update)

    logger.info(
        "Nhận tin nhắn Telegram | conversation_id=%s | message=%s",
        conversation_id,
        user_message,
    )

    try:
        # Báo trạng thái đang xử lý trên Telegram.
        await context.bot.send_chat_action(
            chat_id=update.effective_chat.id,
            action="typing",
        )

        # ChatbotService đang chạy đồng bộ nên đưa sang thread riêng,
        # tránh làm đứng event loop của Telegram Bot.
        response = await asyncio.to_thread(
            call_autobiz,
            user_message,
            conversation_id,
        )

        answer = extract_reply(response)

        if not answer:
            answer = (
                "AutoBiz chưa tạo được câu trả lời phù hợp. "
                "Bạn vui lòng mô tả rõ hơn nhu cầu của mình."
            )

        # Trả lời khách trước.
        await send_long_message(update, answer)

        # Gửi sang n8n khi phát hiện đơn nháp hoặc sự kiện phù hợp.
        if should_send_to_n8n(response):
            payload = build_n8n_payload(
                update=update,
                response=response,
                conversation_id=conversation_id,
                user_message=user_message,
            )

            try:
                await asyncio.to_thread(
                    send_to_n8n,
                    payload,
                )
            except Exception:
                # Lỗi n8n không được làm gián đoạn cuộc trò chuyện
                # hoặc khiến khách nhận thêm thông báo lỗi hệ thống.
                logger.exception(
                    "AutoBiz đã trả lời khách nhưng không gửi được " "dữ liệu sang n8n."
                )

    except Exception:
        logger.exception(
            "Lỗi khi AutoBiz xử lý tin nhắn Telegram | " "conversation_id=%s",
            conversation_id,
        )

        await update.message.reply_text(
            "Hệ thống đang gặp lỗi khi xử lý yêu cầu. " "Bạn vui lòng thử lại."
        )


# =========================================================
# TELEGRAM ERROR HANDLER
# =========================================================


async def error_handler(
    update: object,
    context: ContextTypes.DEFAULT_TYPE,
) -> None:
    """
    Ghi lại các lỗi phát sinh từ Telegram Bot.
    """
    logger.error(
        "Telegram Bot gặp lỗi",
        exc_info=(
            (
                type(context.error),
                context.error,
                context.error.__traceback__,
            )
            if context.error
            else None
        ),
    )


# =========================================================
# APPLICATION ENTRY POINT
# =========================================================


def main() -> None:
    """
    Khởi động Telegram Bot bằng polling.
    """
    if not BOT_TOKEN:
        raise RuntimeError(
            "Không tìm thấy TELEGRAM_BOT_TOKEN.\n"
            "Hãy thêm biến này vào file .env ở thư mục gốc project."
        )

    application = Application.builder().token(BOT_TOKEN).build()

    application.add_handler(
        CommandHandler(
            "start",
            start_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "newchat",
            new_chat_command,
        )
    )

    application.add_handler(
        CommandHandler(
            "reset",
            reset_command,
        )
    )

    application.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    application.add_error_handler(error_handler)

    logger.info("AutoBiz Telegram Bot đang chạy...")
    print("AutoBiz Telegram Bot đang chạy...")

    application.run_polling(
        allowed_updates=Update.ALL_TYPES,
    )


if __name__ == "__main__":
    main()
