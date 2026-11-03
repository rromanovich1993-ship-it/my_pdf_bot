import asyncio
import os
import random
from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import Command
from aiogram.types import (
    ReplyKeyboardMarkup,
    KeyboardButton,
    InlineKeyboardMarkup,
    InlineKeyboardButton,
)
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("bot_token")
VSEGPT_KEY = os.getenv("api_key")

if not TELEGRAM_TOKEN or not VSEGPT_KEY:
    raise ValueError("Не заданы TELEGRAM_BOT_TOKEN или VSEGPT_API_KEY")

ai_client = OpenAI(
    api_key=VSEGPT_KEY,
    base_url="https://api.vsegpt.ru/v1",
)

QUESTIONS = {
    "Python": [
        "В чем разница между списком (list) и кортежем (tuple)?",
        "Что такое декоратор в Python и зачем он нужен?",
        "Объясни разницу между генератором и итератором.",
        "Что такое GIL и как он влияет на многопоточность?",
        "Как работает сборщик мусора в Python?",
    ],
    "Selenium": [
        "В чем разница между неявным и явным ожиданием?",
        "Что такое Page Object Model и какие у него преимущества?",
        "Как в Selenium обрабатывать несколько окон?",
        "Как сделать скриншот при падении теста?",
    ],
    "API": [
        "Чем отличаются методы PUT и PATCH?",
        "Что такое API chaining?",
        "Как автоматизировать тестирование API с помощью Python?",
        "Что такое идемпотентность и почему она важна?",
    ],
    "Pytest": [
        "Что такое фикстуры в pytest и какие у них scope?",
        "Чем отличаются parametrize и fixture?",
        "Как запускать тесты параллельно в pytest?",
    ],
}

# Постоянная клавиатура внизу чата
MAIN_KEYBOARD = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="🐍 Python"), KeyboardButton(text="🌐 Selenium")],
        [KeyboardButton(text="🔗 API"), KeyboardButton(text="🧪 Pytest")],
        [KeyboardButton(text="🎲 Случайный вопрос")],
    ],
    resize_keyboard=True,  # компактный размер
    input_field_placeholder="Выбери тему или нажми кнопку",
)

# Хранилище текущего вопроса по пользователю
user_question_state = {}

bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()


async def generate_detailed_answer(question: str) -> str:
    prompt = f"""
    Ты — эксперт по Python и автоматизации тестирования, готовишь кандидата к собеседованию.

    Вопрос: "{question}"

    Дай ПОДРОБНЫЙ ответ, который можно рассказать на собеседовании. Структура:
    1. Краткое определение (1–2 предложения).
    2. Подробное объяснение с примерами кода (если уместно).
    3. Ключевые нюансы, которые любят спрашивать на собеседованиях.
    4. Частая ошибка или подводный камень.
    5. в конце можно привести пример для понимания не программисту а строителю

    Пиши на русском, структурированно, без воды. Используй markdown умеренно.
    """
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None,
        lambda: ai_client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
        ),
    )
    return response.choices[0].message.content


# --- Обработчики ---

@dp.message(Command("start"))
async def cmd_start(message: types.Message):
    await message.answer(
        "🤖 Привет! Я тренажёр теории для собеседования на Python-автоматизатора.\n\n"
        "Выбери тему на клавиатуре внизу — я задам вопрос, а потом покажу подробный разбор.",
        reply_markup=MAIN_KEYBOARD,
    )


# Обработка нажатия на кнопки тем (текст кнопки = текст сообщения)
@dp.message(F.text.in_({"🐍 Python", "🌐 Selenium", "🔗 API", "🧪 Pytest"}))
async def handle_topic(message: types.Message):
    topic = message.text.split(" ", 1)[1]  # "🐍 Python" -> "Python"
    if topic not in QUESTIONS:
        await message.answer("Тема не найдена.")
        return

    question = random.choice(QUESTIONS[topic])
    user_question_state[message.from_user.id] = question

    # inline-кнопка под вопросом для генерации ответа
    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Показать подробный ответ", callback_data=f"answer_{topic}")],
        [InlineKeyboardButton(text="🔄 Другой вопрос", callback_data=f"another_{topic}")],
    ])

    await message.answer(
        f"❓ **Вопрос по теме «{topic}»:**\n\n{question}",
        reply_markup=answer_keyboard,
        parse_mode="Markdown",
    )


# Кнопка "Случайный вопрос" — выбирает случайную тему и вопрос
@dp.message(F.text == "🎲 Случайный вопрос")
async def handle_random(message: types.Message):
    topic = random.choice(list(QUESTIONS.keys()))
    question = random.choice(QUESTIONS[topic])
    user_question_state[message.from_user.id] = question

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Показать подробный ответ", callback_data=f"answer_{topic}")],
    ])

    await message.answer(
        f"🎲 **Случайный вопрос (тема: {topic}):**\n\n{question}",
        reply_markup=answer_keyboard,
        parse_mode="Markdown",
    )


@dp.callback_query(F.data.startswith("another_"))
async def handle_another(callback: types.CallbackQuery):
    topic = callback.data.split("_", 1)[1]
    question = random.choice(QUESTIONS[topic])
    user_question_state[callback.from_user.id] = question

    answer_keyboard = InlineKeyboardMarkup(inline_keyboard=[
        [InlineKeyboardButton(text="📖 Показать подробный ответ", callback_data=f"answer_{topic}")],
        [InlineKeyboardButton(text="🔄 Другой вопрос", callback_data=f"another_{topic}")],
    ])

    await callback.message.edit_text(
        f"❓ **Вопрос по теме «{topic}»:**\n\n{question}",
        reply_markup=answer_keyboard,
        parse_mode="Markdown",
    )
    await callback.answer()


@dp.callback_query(F.data.startswith("answer_"))
async def handle_answer_request(callback: types.CallbackQuery):
    user_id = callback.from_user.id
    question = user_question_state.get(user_id)

    if not question:
        await callback.answer("Вопрос не найден. Нажми кнопку темы внизу.", show_alert=True)
        return

    await callback.message.edit_text(
        f"❓ **Вопрос:**\n{question}\n\n⏳ Генерирую подробный ответ..."
    )
    await callback.answer()

    try:
        answer = await generate_detailed_answer(question)
        if len(answer) > 4000:
            for i in range(0, len(answer), 4000):
                await callback.message.answer(answer[i:i + 4000])
            await callback.message.edit_text(
                f"❓ **Вопрос:**\n{question}\n\n✅ Ответ отправлен выше (несколько сообщений)."
            )
        else:
            await callback.message.edit_text(
                f"❓ **Вопрос:**\n{question}\n\n📖 **Подробный ответ:**\n\n{answer}",
                parse_mode="Markdown",
            )
    except Exception as e:
        await callback.message.edit_text(f"❌ Ошибка при генерации ответа: {e}")


# Ловим любой другой текст, чтобы не молчать
@dp.message(F.text)
async def fallback(message: types.Message):
    await message.answer(
        "Выбери тему кнопкой внизу 👇",
        reply_markup=MAIN_KEYBOARD,
    )


async def main():
    print("Бот запущен. Нажмите Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())

#python tg_agent.py  это для запуска бота