import os
import pandas as pd
import asyncio
from datetime import datetime
from telethon import TelegramClient
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# 1. Load the secret vault
load_dotenv()
API_ID = os.getenv('TELEGRAM_API_ID')
API_HASH = os.getenv('TELEGRAM_API_HASH')
PHONE_NUMBER = os.getenv('TELEGRAM_PHONE')

# 2. Define the OSINT Targets
# These are public channels. We capture the raw Russian, Ukrainian, and Middle Eastern feeds.
TARGET_CHANNELS = [
    {"username": "rt_russian", "perspective": "Russia (Raw)", "lang": "ru"},
    {"username": "V_Zelenskiy_official", "perspective": "Ukraine (Raw)", "lang": "uk"},
    {"username": "AJA_Egypt", "perspective": "Middle East (Raw)", "lang": "ar"}
]


def translate_text(text, source_lang):
    if source_lang == 'en' or not text:
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception:
        return text


async def fetch_telegram_data():
    all_messages = []
    print("Starting the Telegram OSINT Siphon...")

    # Initialize the client. This creates a local 'truth_tide.session' file.
    client = TelegramClient('truth_tide', API_ID, API_HASH)
    await client.start(phone=PHONE_NUMBER)

    for channel in TARGET_CHANNELS:
        print(f" -> Siphoning raw data from @{channel['username']}...")
        try:
            # Fetch the last 15 text messages from the channel
            async for msg in client.iter_messages(channel['username'], limit=15):
                if msg.text:
                    # Clean the text (Telegram posts have lots of messy line breaks)
                    raw_text = msg.text.replace('\n', ' ')
                    english_text = translate_text(raw_text, channel['lang'])

                    # Telegram posts don't have "headlines", so we use the first 150 characters
                    article = {
                        "source": f"Telegram: @{channel['username']}",
                        "perspective": channel['perspective'],
                        "original_language": channel['lang'],
                        "raw_title": raw_text[:150] + "...",
                        "title": english_text[:150] + "...",
                        "published_date": msg.date.strftime("%Y-%m-%d %H:%M:%S"),
                        "link": f"https://t.me/{channel['username']}/{msg.id}",
                        "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }
                    all_messages.append(article)
        except Exception as e:
            print(f"Failed to scrape {channel['username']}: {e}")

    await client.disconnect()
    return all_messages


def save_to_csv(articles_list):
    if not articles_list:
        print("No data collected.")
        return

    df = pd.DataFrame(articles_list)
    today_str = datetime.now().strftime("%Y_%m_%d")
    # Save it to the raw folder so our AI engine can pick it up!
    filename = f"data/raw/osint_telegram_{today_str}.csv"

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"\nSuccess! Saved {len(df)} Telegram posts to {filename}")


if __name__ == "__main__":
    # Telethon requires an asynchronous event loop
    loop = asyncio.get_event_loop()
    scraped_data = loop.run_until_complete(fetch_telegram_data())
    save_to_csv(scraped_data)