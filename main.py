import asyncio
import logging
import sys
from os import getenv

from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram.types import Message

# Загружаем токен из .env
load_dotenv()
TOKEN = getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)

# Диспетчер — маршрутизатор сообщений
bot = Bot(token=TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher()

@dp.message(CommandStart())
async def command_start_handler(message: Message) -> None:
    """Обработчик команды /start."""
    await message.answer(f"Привет, {html.bold(message.from_user.full_name)}! Я готов к работе.")

@dp.message()
async def echo_handler(message: Message) -> None:
    """Эхо-обработчик: повторяет текст, на остальное отвечает заглушкой."""
    try:
        await message.send_copy(chat_id=message.chat.id)
    except TypeError:
        await message.answer("Пока я умею только повторять текст. PDF добавим позже.")

async def main() -> None:
    bot = Bot(
        token=TOKEN,
        default=DefaultBotProperties(parse_mode=ParseMode.HTML)
    )
    await dp.start_polling(bot)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())