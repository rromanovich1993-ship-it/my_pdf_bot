import os
import asyncio
import json
import logging
import re
import fitz  # pymupdf
import pandas as pd
import base64
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message, FSInputFile
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

load_dotenv()

TG_TOKEN = os.getenv("")
VSEGPT_KEY = os.getenv("")

if not TG_TOKEN or not VSEGPT_KEY:
    raise ValueError("❌ Проверьте .env! Нужны BOT_TOKEN и API_KEY.")

MODEL_NAME = "openai/gpt-4o-mini"

bot = Bot(token=TG_TOKEN)
dp = Dispatcher()

SYSTEM_PROMPT = (
    "Ты — ведущий инженер ПТО и эксперт-сметчик по железобетонным конструкциям (КЖ).\n"
    "Твоя задача — извлечь ведомость объемов материалов из текста чертежа.\n\n"
    "Сформируй ответ строго в формате валидного JSON. Используй только стандартные двойные кавычки для ключей и строк.\n"
    "Шаблон ответа:\n"
    "{\n"
    '  "beton": [{"element": "Плита Пм1", "class": "B25", "volume_m3": 12.5}],\n'
    '  "armatura": [{"class": "A500C", "diameter_mm": 12, "weight_kg": 450.8}]\n'
    "}\n"
    "Не пиши никаких пояснений, комментариев, markdown-тегов и лишнего текста вокруг JSON."
)


def extract_text_from_pdf(pdf_path: str) -> str:
    """Извлекает текст из PDF"""
    doc = fitz.open(pdf_path)
    parts = []
    for i, page in enumerate(doc):
        parts.append(page.get_text("text"))
    doc.close()
    return "".join(parts)


def pdf_pages_to_base64_images(pdf_path: str):
    """Конвертирует первые 5 страниц PDF в картинки base64"""
    doc = fitz.open(pdf_path)
    base64_images = []
    for page_num in range(min(5, len(doc))):
        page = doc[page_num]
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5))
        img_data = pix.tobytes("png")
        base64_str = base64.b64encode(img_data).decode("utf-8")
        base64_images.append(base64_str)
    doc.close()
    return base64_images


def _request_vsegpt(messages: list) -> str:
    """Надежный прямой HTTP-запрос к VseGPT без использования сбоящих сторонних библиотек"""
    url = "https://vsegpt.ru"
    headers = {
        "Authorization": f"Bearer {VSEGPT_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": messages,
        "temperature": 0.5,
        "response_format": {"type": "json_object"}
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=180)
        if response.status_code == 200:
            res_json = response.json()
            return res_json["choices"][0]["message"]["content"]
        else:
            return json.dumps({"error": f"Код сервера {response.status_code}: {response.text}"}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def analyze_text_method(extracted_text: str) -> str:
    """Метод 1: Быстрый текстовый анализ"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Вот текст чертежа:\n\n{extracted_text[:100_000]}"}
    ]
    # Запускаем синхронный запрос в пуле потоков
    return _request_vsegpt(messages)


def analyze_vision_method(base64_images):
    """Метод 2: Резервный визуальный анализ (Vision)"""
    content_list = [{"type": "text", "text": SYSTEM_PROMPT}]
    for b64_img in base64_images:
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{b64_img}"}
        })
    messages = [{"role": "user", "content": content_list}]
    return _request_vsegpt(messages)


def _clean_and_parse_json(raw: str) -> dict:
    """Очищает строку от мусора и преобразует в словарь"""
    if not raw:
        return {"beton": [], "armatura": []}

    text = raw.strip()

    if "<html" in text.lower() or "<doctype" in text.lower():
        log.error("Вместо JSON пришла HTML страница ошибки.")
        return {"beton": [], "armatura": []}

    match = re.search(r'\{.*\}', text, re.DOTALL)
    if not match:
        return {"beton": [], "armatura": []}

    text = match.group(0).strip()

    text = re.sub(r'[“”„“]', '"', text)
    text = re.sub(r"\'", '"', text)

    try:
        return json.loads(text)
    except Exception as e:
        log.error(f"Ошибка парсинга JSON: {e}")
        return {"beton": [], "armatura": []}


def save_to_excel(data: dict, output_path: str):
    """Сохраняет структуру данных в Excel файл"""
    df_beton = pd.DataFrame(data.get("beton", []))
    df_arm = pd.DataFrame(data.get("armatura", []))
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        if not df_beton.empty:
            df_beton.to_excel(writer, sheet_name="Объемы бетона", index=False)
        else:
            pd.DataFrame([{"Сообщение": "Данные о бетоне не найдены"}]).to_excel(writer, sheet_name="Объемы бетона",
                                                                                 index=False)

        if not df_arm.empty:
            df_arm.to_excel(writer, sheet_name="Расход арматуры", index=False)
        else:
            pd.DataFrame([{"Сообщение": "Данные об арматуре не найдены"}]).to_excel(writer,
                                                                                    sheet_name="Расход арматуры",
                                                                                    index=False)


@dp.message(F.text == "/start")
async def cmd_start(message: Message):
    await message.answer(
        "🏗️ **Привет! Я гибридный ИИ-агент ПТО.**\n\n"
        "Отправь мне файл чертежа КЖ. Я соберу спецификации в Excel."
    )


@dp.message(F.document)
async def handle_pdf(message: Message):
    if not (message.document.file_name or "").lower().endswith(".pdf"):
        await message.answer("❌ Пришли файл в формате PDF.")
        return

    status_msg = await message.answer("⏳ Шаг 1: Проверяю текстовый слой чертежа...")

    os.makedirs("downloads", exist_ok=True)
    file_base_name = message.document.file_name.rsplit(".", 1)[0]
    pdf_path = os.path.join("downloads", f"{message.document.file_id}.pdf")
    excel_path = os.path.join("downloads", f"Выгрузка_{file_base_name}.xlsx")

    try:
        # 1. Скачивание
        file_info = await bot.get_file(message.document.file_id)
        await bot.download_file(file_info.file_path, pdf_path)

        # 2. Текстовый метод
        text_data = extract_text_from_pdf(pdf_path)
        parsed_data = {"beton": [], "armatura": []}

        if len(text_data.strip()) > 200:
            await status_msg.edit_text("🧠 Текст найден. Анализирую спецификации быстрым методом...")
            loop = asyncio.get_event_loop()
            ai_response = await loop.run_in_executor(None, analyze_text_method, text_data)
            parsed_data = _clean_and_parse_json(ai_response)

        # 3. Визуальный метод (Vision)
        if not parsed_data.get("beton") and not parsed_data.get("armatura"):
            await status_msg.edit_text("🔄 Включаю ИИ-зрение. Оптимизирую и сканирую листы чертежа...")
            images = pdf_pages_to_base64_images(pdf_path)

            await status_msg.edit_text(f"👁️ Сканирую глазами {len(images)} листов. Распознаю таблицы КЖ...")
            loop = asyncio.get_event_loop()
            ai_response = await loop.run_in_executor(None, analyze_vision_method, images)
            parsed_data = _clean_and_parse_json(ai_response)

        if not parsed_data.get("beton") and not parsed_data.get("armatura"):
            await status_msg.edit_text("❌ К сожалению, ИИ не смог обнаружить таблицы КЖ на листах документа.")
            return

        await status_msg.edit_text("📊 Спецификации успешно распознаны! Формирую Excel-отчет...")
        save_to_excel(parsed_data, excel_path)

        await status_msg.delete()
        await message.answer_document(FSInputFile(excel_path), caption="📊 Готово! Объемы КЖ собраны гибридным методом.")

    except Exception as e:
        log.exception("Ошибка в гибридном обработчике")
        await status_msg.edit_text(f"❌ Ошибка агента: {e}")
    finally:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)


async def main():
    print("🤖 Полноценный гибридный Бот-ПТО успешно запущен в IDEA!")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())




