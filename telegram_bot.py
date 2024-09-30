# telegram_bot.py
import os
import logging
import aiohttp
import sys
import asyncio
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
from aiogram.utils import executor
from dotenv import load_dotenv
from urllib.parse import parse_qs, urlencode
from urllib.parse import urljoin
import json

# Load environment variables
load_dotenv()

# Get the chat name from command line argument
if len(sys.argv) > 1:
    CHAT_NAME = sys.argv[1]
else:
    print("Please provide a chat name as a command line argument.")
    sys.exit(1)

# Construct dynamic environment variable name for Telegram token
TELEGRAM_TOKEN_VAR = f"TELEGRAM_TOKEN_{CHAT_NAME.upper()}"
STARTER_MESSAGE_VAR = f"STARTER_MESSAGE_{CHAT_NAME.upper()}"

# Get the Telegram bot token and FastAPI URL from environment variables
TELEGRAM_TOKEN = os.getenv(TELEGRAM_TOKEN_VAR)
FASTAPI_URL = os.getenv('FASTAPI_URL', 'https://abcdefg-fastapi.onrender.com/chat')
STARTER_MESSAGE = os.getenv(STARTER_MESSAGE_VAR, "Welcome to the chat!")

# Define the DISCLAIMER constant
DISCLAIMER = "Disclaimer: This AI Chatbot can make mistakes. Please verify the information. This chatbot is intended for educational and informational purposes only."

if not TELEGRAM_TOKEN:
    raise ValueError(f"No {TELEGRAM_TOKEN_VAR} found in environment variables")

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Initialize a global aiohttp session
session = None

# Handler for /start command
@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await message.reply(f"Hello! {STARTER_MESSAGE}")

# Handler for /help command
@dp.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    await message.reply("Just type your question and I will try to answer.")

MAX_MESSAGE_LENGTH = 4000

# Handler for text messages
@dp.message_handler(lambda message: message.text and not message.text.startswith('/'))
async def handle_message(message: types.Message):
    user_message = message.text
    payload = {"name": CHAT_NAME, "prompt": user_message}

    try:
        # Send a typing action
        await bot.send_chat_action(message.chat.id, 'typing')

        # Send an initial "thinking" message
        thinking_message = await message.reply("Thinking...")

        async with session.post(FASTAPI_URL, params=payload, timeout=60) as resp:
            if resp.status == 200:
                full_response = ""
                first_chunk = True
                source_info = None
                async for chunk in resp.content.iter_any():
                    decoded_chunk = chunk.decode('utf-8')
                    
                    # Check if the chunk is the source information JSON
                    if decoded_chunk.startswith('\n{') and decoded_chunk.endswith('}'):
                        source_info = json.loads(decoded_chunk.strip())
                        break
                    
                    full_response += decoded_chunk

                    if len(full_response) > MAX_MESSAGE_LENGTH:
                        full_response = full_response[:MAX_MESSAGE_LENGTH] + "... (truncated)"
                        break

                    if first_chunk:
                        # Replace "Thinking..." with the first part of the response
                        await thinking_message.edit_text(full_response, parse_mode=ParseMode.HTML)
                        first_chunk = False
                    elif len(full_response) % 100 == 0 or DISCLAIMER in full_response:
                        # Update the message periodically or when we receive the disclaimer
                        await thinking_message.edit_text(full_response, parse_mode=ParseMode.HTML)
                        await asyncio.sleep(0.1)  # Add a small delay to avoid rate limiting

                # Create inline keyboard with source buttons
                keyboard = InlineKeyboardMarkup()
                if source_info:
                    if source_info["source1"]:
                        keyboard.add(InlineKeyboardButton(source_info["source1"]["title"], url=source_info["source1"]["url"]))
                    if source_info["source2"]:
                        keyboard.add(InlineKeyboardButton(source_info["source2"]["title"], url=source_info["source2"]["url"]))

                # Final update to ensure we've sent everything
                if full_response != thinking_message.text:
                    await thinking_message.edit_text(full_response, parse_mode=ParseMode.HTML, reply_markup=keyboard)

            else:
                await thinking_message.edit_text('Sorry, there was an error processing your request.')
                error_text = await resp.text()
                logger.error(f"Error {resp.status}: {error_text}")
    except aiohttp.ClientError as e:
        await message.reply('An error occurred while connecting to the server.')
        logger.error(f"HTTP Client Error: {e}")
    except Exception as e:
        await message.reply('An unexpected error occurred. Please try again later.')
        logger.error(f"Unexpected Exception: {e}")

# Handler for unknown commands
@dp.message_handler()
async def unknown_command(message: types.Message):
    await message.reply("Sorry, I didn't understand that command.")

async def on_startup(dp: Dispatcher):
    global session
    session = aiohttp.ClientSession()
    logger.info(f"Bot started! CHAT_NAME: {CHAT_NAME}, Token Variable: {TELEGRAM_TOKEN_VAR}")

async def on_shutdown(dp: Dispatcher):
    await session.close()  # Close the aiohttp session
    await bot.close()
    logger.info("Bot and session shutdown!")

def main():
    executor.start_polling(
        dp,
        skip_updates=True,
        on_startup=on_startup,
        on_shutdown=on_shutdown
    )

if __name__ == "__main__":
    main()
