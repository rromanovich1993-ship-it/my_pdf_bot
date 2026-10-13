import os
import json
import logging
import asyncio
import httpx  # Используем напрямую вместо библиотеки openai
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from dotenv import load_dotenv

load_dotenv()

API_TOKEN = os.getenv('BOT_TOKEN')
VSE_GPT_KEY = os.getenv('API_KEY')  # Ваш ключ VseGPT (sk-or-v1-...)

logging.basicConfig(level=logging.INFO)

bot = Bot(token=API_TOKEN)
dp = Dispatcher()

# Правильное имя модели и URL для прямого API VseGPT
MODEL_NAME = "openai/gpt-4o-mini"
VSE_GPT_URL = "https://vsegpt.ru"

ALLOWED_OBJECTS = ["ЖК 'Северный', Корпус 1", "БЦ 'Орион'"]
ALLOWED_WORKS = ["Бетонирование плиты", "Армирование стен", "Кладка кирпича"]

SYSTEM_PROMPT = f"""
Ты — ассистент по планированию в строительстве (работаешь в связке с Primavera). 
Твоя задача — получить текстовый отчет от прораба и извлечь из него 3 сущности в формате JSON:
1. "object" — строительный объект. Обязательно сопоставь его со списком: {ALLOWED_OBJECTS}. Если в тексте похожий объект, выбери точное название из списка. Если объекта нет или он вообще не подходит, запиши null.
2. "work" — вид работы. Обязательно сопоставь со списком: {ALLOWED_WORKS}. Если работы нет или она не подходит, запиши null.
3. "volume" — объем выполненных работ. Вытащи только число (float). Если число не указано, запиши null.

Отвечай СТРОГО в формате JSON без какого-либо лишнего текста, форматирования или кавычек ```json.

Пример ответа:
{{"object": "ЖК 'Северный', Корпус 1", "work": "Бетонирование плиты", "volume": 25.5}}
"""


@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "Приветствую! Я ИИ-ассистент сбора факта для Primavera.\n\n"
        "Просто напишите мне, что сделано на объекте в свободной форме.\n"
        "Пример: 'На северном корпусе 1 залили примерно 12.5 кубов на плиту перекрытия'"
    )


@dp.message()
async def process_ai_fact(message: types.Message):
    await bot.send_chat_action(chat_id=message.chat.id, action="typing")
    waiting_message = await message.answer("🤖 ИИ анализирует ваше сообщение...")

    # Формируем заголовки и тело запроса по спецификации VseGPT / OpenAI
    headers = {
        "Authorization": f"Bearer {VSE_GPT_KEY}",
        "Content-Type": "application/json",
        "X-Title": "Telegram Stroy Bot"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": message.text}
        ],
        "temperature": 0.0
    }

    try:
        # Делаем асинхронный POST-запрос напрямую к VseGPT
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(VSE_GPT_URL, headers=headers, json=payload)

            # Проверяем статус ответа (если не 200, вызовется исключение)
            response.raise_for_status()

            # Парсим ответ сервера
            response_json = response.json()

        # Извлекаем текст ответа ИИ из стандартной структуры OpenAI API
        ai_response_text = response_json['choices'][0]['message']['content'].strip()

        # Очищаем от возможных markdown-оберток нейросети
        ai_response_text = ai_response_text.replace("```json", "").replace("```", "").strip()

        # Парсим полученную JSON-строку с данными
        data = json.loads(ai_response_text)

        obj = data.get("object")
        work = data.get("work")
        volume = data.get("volume")

        # Навыки QA: Валидация данных
        if not obj or not work or not volume:
            missing = []
            if not obj: missing.append("Объект")
            if not work: missing.append("Работа")
            if not volume: missing.append("Объем")

            await waiting_message.edit_text(
                f"❌ Не удалось распознать все данные.\n"
                f"Не определено: *{', '.join(missing)}*.\n"
                f"Попробуйте написать точнее (укажите объект, работу и число объема).",
                parse_mode="Markdown"
            )
            return

        # Итоговая карточка
        report = (
            f"✅ *ИИ успешно распознал факт!*\n\n"
            f"🏗 *Объект:* {obj}\n"
            f"🛠 *Работа:* {work}\n"
            f"📊 *Объем:* {volume}\n\n"
            f"Данные готовы для выгрузки в Primavera."
        )
        await waiting_message.edit_text(report, parse_mode="Markdown")
        print(f"ИИ СОХРАНИЛ: {obj} | {work} | {volume}")

    except httpx.HTTPStatusError as http_err:
        logging.error(f"Ошибка HTTP от VseGPT: {http_err.response.status_code} - {http_err.response.text}")
        await waiting_message.edit_text(
            f"❌ Ошибка сервера нейросети ({http_err.response.status_code}). Проверьте баланс VseGPT.")
    except (json.JSONDecodeError, KeyError, IndexError) as parse_err:
        logging.error(f"Ошибка парсинга ответа: {parse_err}")
        await waiting_message.edit_text("❌ Ошибка обработки данных ИИ. Попробуйте написать по-другому.")
    except Exception as e:
        logging.error(f"Неизвестная ошибка: {e}", exc_info=True)
        await waiting_message.edit_text("Произошла ошибка при обработке сообщения. Попробуйте позже.")


async def main():
    await dp.start_polling(bot)


if __name__ == '__main__':
    asyncio.run(main())