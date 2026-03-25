import feedparser
import pandas as pd
import os
from datetime import datetime
from deep_translator import GoogleTranslator

# 1. Expanded Targets (Multilingual)
NEWS_FEEDS = [
    {"source": "BBC News", "url": "http://feeds.bbci.co.uk/news/world/rss.xml", "perspective": "Western Europe",
     "lang": "en"},
    {"source": "Al Jazeera (EN)", "url": "https://www.aljazeera.com/xml/rss/all.xml",
     "perspective": "Middle East (English)", "lang": "en"},
    {"source": "Al Jazeera (AR)",
     "url": "https://www.aljazeera.net/aljazeerarss/a7c186be-1baa-4bd4-9d80-a84db769f779/73d0e1b4-532f-45ef-b135-bfdff8b8cab9",
     "perspective": "Middle East (Arabic)", "lang": "ar"},
    {"source": "Lenta (RU)", "url": "https://lenta.ru/rss/news", "perspective": "Russia", "lang": "ru"}
]


def translate_text(text, source_lang):
    """Translates text to English if it's not already in English."""
    if source_lang == 'en' or not text:
        return text
    try:
        # Translate to english
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        return translated
    except Exception as e:
        print(f"Translation error: {e}")
        return text


def fetch_news_data(feeds):
    all_articles = []
    print("Starting the Multilingual Data Siphon...")

    for feed in feeds:
        print(f" -> Fetching and translating data from {feed['source']}...")
        parsed_feed = feedparser.parse(feed['url'])

        for entry in parsed_feed.entries[:15]:  # Grab top 15 to save time
            # Get raw text
            raw_title = entry.get("title", "No Title")
            raw_summary = entry.get("summary", "No Summary")

            # Translate text
            english_title = translate_text(raw_title, feed["lang"])

            article = {
                "source": feed["source"],
                "perspective": feed["perspective"],
                "original_language": feed["lang"],
                "raw_title": raw_title,
                "title": english_title,  # Our AI will analyze this translated title
                "published_date": entry.get("published", "No Date"),
                "link": entry.get("link", "No Link"),
                "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_articles.append(article)

    return all_articles


def save_to_csv(articles_list):
    df = pd.DataFrame(articles_list)
    today_str = datetime.now().strftime("%Y_%m_%d")
    filename = "data/raw/global_headlines_latest.csv"
    df.to_csv(filename, index=False)
    print(f"\nSuccess! Saved {len(df)} translated articles to {filename}")

    print("\n--- Translation Preview ---")
    # Show the source, the original language, and the English translation
    print(df[df['original_language'] != 'en'][['source', 'raw_title', 'title']].head(5))


if __name__ == "__main__":
    scraped_data = fetch_news_data(NEWS_FEEDS)
    save_to_csv(scraped_data)