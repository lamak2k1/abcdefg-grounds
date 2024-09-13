# telegram_bot.py
import os
import logging
import aiohttp
from aiogram import Bot, Dispatcher, types
from aiogram.contrib.middlewares.logging import LoggingMiddleware
from aiogram.types import InlineKeyboardMarkup, InlineKeyboardButton, ParseMode
from aiogram.utils import executor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Enable logging
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get the Telegram bot token and FastAPI URL from environment variables
TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN')
FASTAPI_URL = os.getenv('FASTAPI_URL', 'https://abcdefg-fastapi.onrender.com/chat')

if not TELEGRAM_TOKEN:
    raise ValueError("No TELEGRAM_TOKEN found in environment variables")

# Initialize bot and dispatcher
bot = Bot(token=TELEGRAM_TOKEN, parse_mode=ParseMode.HTML)
dp = Dispatcher(bot)
dp.middleware.setup(LoggingMiddleware())

# Initialize a global aiohttp session
session = aiohttp.ClientSession()

# Handler for /start command
@dp.message_handler(commands=['start'])
async def cmd_start(message: types.Message):
    await message.reply("Hello! Ask me anything on stock market investing.")

# Handler for /help command
@dp.message_handler(commands=['help'])
async def cmd_help(message: types.Message):
    await message.reply("Just type your question and I will try to answer.")

# Handler for text messages
@dp.message_handler(lambda message: message.text and not message.text.startswith('/'))
async def handle_message(message: types.Message):
    user_message = message.text
    name = "karl"  # You can make this dynamic if needed
    payload = {"name": name, "prompt": user_message}

    try:
        # Use the global session
        async with session.post(FASTAPI_URL, params=payload) as resp:
            if resp.status == 200:
                data = await resp.json()
                answer = data.get('answer', 'Sorry, I could not understand your question.')

                # Initialize keyboard as None
                keyboard = None

                # Extract titles, sources, and scores from the response
                title1 = data.get('title1')
                source1 = data.get('source1')
                score1 = data.get('score1', 0)
                
                title2 = data.get('title2')
                source2 = data.get('source2')
                score2 = data.get('score2', 0)

                # Check if both scores are greater than 0.7 and all necessary fields are present
                if (score1 > 0.7 and score2 > 0.7 and
                    title1 and source1 and title2 and source2):
                    
                    # Create inline keyboard with two buttons
                    keyboard = InlineKeyboardMarkup(row_width=2).add(
                        InlineKeyboardButton(text=title1.strip('"'), url=source1),
                        InlineKeyboardButton(text=title2.strip('"'), url=source2)
                    )

                # Reply with answer and conditional keyboard
                if keyboard:
                    await message.reply(answer, reply_markup=keyboard)
                else:
                    await message.reply(answer)

            else:
                await message.reply('Sorry, there was an error processing your request.')
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
    logger.info("Bot started!")

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