import feedparser
import pandas as pd
import os
import requests
from bs4 import BeautifulSoup
from datetime import datetime
from deep_translator import GoogleTranslator

# ─────────────────────────────────────────────
# 1. EXPANDED NEWS SOURCES (14 sources, 8 languages)
# ─────────────────────────────────────────────
NEWS_FEEDS = [
    # --- Western / NATO Perspective ---
    {"source": "BBC News",
     "url": "http://feeds.bbci.co.uk/news/world/rss.xml",
     "perspective": "Western Europe", "lang": "en"},

    {"source": "Reuters",
     "url": "https://feeds.reuters.com/reuters/worldNews",
     "perspective": "Western (Wire)", "lang": "en"},

    {"source": "Deutsche Welle (EN)",
     "url": "https://rss.dw.com/rdf/rss-en-world",
     "perspective": "Western Europe", "lang": "en"},

    {"source": "France 24 (EN)",
     "url": "https://www.france24.com/en/rss",
     "perspective": "Western Europe", "lang": "en"},

    # --- Middle East Perspective ---
    {"source": "Al Jazeera (EN)",
     "url": "https://www.aljazeera.com/xml/rss/all.xml",
     "perspective": "Middle East (English)", "lang": "en"},

    {"source": "Al Jazeera (AR)",
     "url": "https://www.aljazeera.net/aljazeerarss/a7c186be-1baa-4bd4-9d80-a84db769f779/73d0e1b4-532f-45ef-b135-bfdff8b8cab9",
     "perspective": "Middle East (Arabic)", "lang": "ar"},

    {"source": "Al Arabiya (EN)",
     "url": "https://www.alarabiya.net/tools/rss",
     "perspective": "Middle East (Gulf)", "lang": "en"},

    {"source": "Tehran Times",
     "url": "https://www.tehrantimes.com/rss",
     "perspective": "Iran (State)", "lang": "en"},

    # --- Russian / Eurasian Perspective ---
    {"source": "Lenta (RU)",
     "url": "https://lenta.ru/rss/news",
     "perspective": "Russia", "lang": "ru"},

    {"source": "RT (EN)",
     "url": "https://www.rt.com/rss/news/",
     "perspective": "Russia (English)", "lang": "en"},

    {"source": "TASS (EN)",
     "url": "https://tass.com/rss/v2.xml",
     "perspective": "Russia (Wire)", "lang": "en"},

    # --- Asian Perspective ---
    {"source": "CGTN (EN)",
     "url": "https://www.cgtn.com/subscribe/rss/section/world.do",
     "perspective": "China (State)", "lang": "en"},

    {"source": "The Hindu",
     "url": "https://www.thehindu.com/news/international/feeder/default.rss",
     "perspective": "South Asia", "lang": "en"},

    # --- Global / Alternative ---
    {"source": "Middle East Eye",
     "url": "https://www.middleeasteye.net/rss",
     "perspective": "Middle East (Independent)", "lang": "en"},

    # --- North American Perspective ---
        {"source": "CNN (World)",
         "url": "http://rss.cnn.com/rss/edition_world.rss",
         "perspective": "North America (Mainstream)", "lang": "en"},

        {"source": "Fox News (World)",
         "url": "http://feeds.foxnews.com/foxnews/world",
         "perspective": "North America (Conservative)", "lang": "en"},

    # --- Indian / South Asian Perspective ---
        {"source": "WION",
         "url": "https://www.wionews.com/feeds/world-news.xml",
         "perspective": "India (International/Geopolitical)", "lang": "en"},

        {"source": "Times of India",
         "url": "https://timesofindia.indiatimes.com/rssfeeds/296589292.cms",
         "perspective": "India (Mainstream)", "lang": "en"},

        {"source": "The Indian Express",
         "url": "https://indianexpress.com/section/world/feed/",
         "perspective": "India (Center-Left)", "lang": "en"},

        {"source": "Hindustan Times",
         "url": "https://www.hindustantimes.com/feeds/rss/world-news/rssfeed.xml",
         "perspective": "India (Mainstream)", "lang": "en"},

        {"source": "Firstpost",
         "url": "https://www.firstpost.com/rss/world.xml",
         "perspective": "India (Center-Right)", "lang": "en"},
]


# ─────────────────────────────────────────────
# 2. TRANSLATION
# ─────────────────────────────────────────────
def translate_text(text, source_lang):
    """Translates text to English. Caps at 800 chars to avoid API limits."""
    if source_lang == 'en' or not text:
        return text
    try:
        # Deep translator has a char limit — chunk if needed
        text = str(text)[:800]
        translated = GoogleTranslator(source=source_lang, target='en').translate(text)
        return translated
    except Exception as e:
        print(f"  Translation error: {e}")
        return text


# ─────────────────────────────────────────────
# 3. FULL ARTICLE SCRAPER
# ─────────────────────────────────────────────
def scrape_article_body(url: str, max_chars: int = 1500) -> str:
    """
    Fetches the full article text from a URL.
    Returns first 1500 chars — intro paragraphs carry the most framing bias.
    Falls back to empty string on any failure (paywall, timeout, bot block).
    """
    if not url or url == "No Link":
        return ""

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/120.0.0.0 Safari/537.36"
        }
        response = requests.get(url, timeout=8, headers=headers)

        if response.status_code != 200:
            return ""

        soup = BeautifulSoup(response.text, "html.parser")

        # Remove junk tags that pollute the text
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form", "figure",
                         "figcaption", "iframe", "noscript"]):
            tag.decompose()

        # Try article-specific tags first (cleaner text)
        article_tag = soup.find("article")
        if article_tag:
            paragraphs = article_tag.find_all("p")
        else:
            paragraphs = soup.find_all("p")

        body = " ".join(p.get_text(strip=True) for p in paragraphs)

        # Clean up whitespace
        body = " ".join(body.split())

        return body[:max_chars]

    except Exception as e:
        print(f"  Scrape failed ({url[:50]}...): {e}")
        return ""


# ─────────────────────────────────────────────
# 4. MAIN FETCH FUNCTION
# ─────────────────────────────────────────────
def fetch_news_data(feeds):
    all_articles = []
    print("Starting the Multilingual Data Siphon (Full Article Mode)...")

    for feed in feeds:
        print(f"\n -> [{feed['source']}] Fetching...")
        try:
            parsed_feed = feedparser.parse(feed['url'])
        except Exception as e:
            print(f"  Feed parse failed: {e}")
            continue

        count = 0
        for entry in parsed_feed.entries[:40]:  # Top 15 per source
            raw_title   = entry.get("title", "").strip()
            raw_summary = entry.get("summary", "").strip()
            link        = entry.get("link", "")

            if not raw_title:
                continue

            # ── Translate title ──
            english_title = translate_text(raw_title, feed["lang"])

            # ── Translate RSS summary (free, no scraping needed) ──
            english_summary = translate_text(raw_summary, feed["lang"]) if raw_summary else ""

            # ── Scrape full article body ──
            print(f"    Scraping: {raw_title[:60]}...")
            raw_body      = scrape_article_body(link)
            english_body  = translate_text(raw_body, feed["lang"]) if raw_body else ""

            # ── Build full_text: title + summary + body ──
            # This is what the AI will analyze — rich, multi-sentence context
            parts     = [p for p in [english_title, english_summary, english_body] if p]
            full_text = " ".join(parts)[:2000]  # Cap at 2000 chars for the AI model

            article = {
                "source":            feed["source"],
                "perspective":       feed["perspective"],
                "original_language": feed["lang"],
                "raw_title":         raw_title,
                "title":             english_title,
                "full_text":         full_text,       # ← AI analyzes THIS
                "published_date":    entry.get("published", "No Date"),
                "link":              link,
                "fetch_timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            all_articles.append(article)
            count += 1

        print(f"    Done. {count} articles collected from {feed['source']}.")

    return all_articles


# ─────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────
def save_to_csv(articles_list):
    if not articles_list:
        print("No articles collected.")
        return

    df = pd.DataFrame(articles_list)
    os.makedirs("data/raw", exist_ok=True)
    filename = "data/raw/global_headlines_latest.csv"
    df.to_csv(filename, index=False)

    print(f"\nSuccess! Saved {len(df)} articles to {filename}")
    print(f"Sources collected: {df['source'].value_counts().to_dict()}")

    print("\n--- Translation + Body Preview ---")
    sample = df[df['original_language'] != 'en'][['source', 'raw_title', 'title', 'full_text']].head(3)
    for _, row in sample.iterrows():
        print(f"\nSource:    {row['source']}")
        print(f"Original:  {row['raw_title'][:80]}")
        print(f"Translated:{row['title'][:80]}")
        print(f"Full Text: {row['full_text'][:150]}...")


if __name__ == "__main__":
    scraped_data = fetch_news_data(NEWS_FEEDS)
    save_to_csv(scraped_data)