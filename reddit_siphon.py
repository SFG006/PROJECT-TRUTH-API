import os
import requests
import pandas as pd
import praw
from bs4 import BeautifulSoup
from datetime import datetime
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# ─────────────────────────────────────────────
# 1. LOAD CREDENTIALS
# ─────────────────────────────────────────────
load_dotenv()
CLIENT_ID     = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_SECRET')
USER_AGENT    = os.getenv('REDDIT_USER_AGENT')

# ─────────────────────────────────────────────
# 2. EXPANDED SUBREDDITS (12 sources, 4 categories)
# ─────────────────────────────────────────────
TARGET_SUBS = [
    # --- Global Mainstream News ---
    {"subreddit": "worldnews",
     "perspective": "Global (Mainstream)", "lang": "en"},

    {"subreddit": "geopolitics",
     "perspective": "Global (Analytical)", "lang": "en"},

    {"subreddit": "GlobalNews",
     "perspective": "Global (Crowd)", "lang": "en"},

    # --- Conflict-Specific ---
    {"subreddit": "ukraine",
     "perspective": "Ukraine (Regional)", "lang": "en"},

    {"subreddit": "UkraineRussiaReport",
     "perspective": "Ukraine/Russia (Neutral Reports)", "lang": "en"},

    {"subreddit": "russia",
     "perspective": "Russia (Crowd)", "lang": "en"},

    # --- Middle East ---
    {"subreddit": "AskMiddleEast",
     "perspective": "Middle East (Crowd)", "lang": "en"},

    {"subreddit": "israelpalestine",
     "perspective": "Middle East (Contested)", "lang": "en"},

    {"subreddit": "iranian",
     "perspective": "Iran (Diaspora)", "lang": "en"},

    # --- Asia & Global South ---
    {"subreddit": "GenZedong",
     "perspective": "China (Sympathetic)", "lang": "en"},

    {"subreddit": "Sino",
     "perspective": "China (Crowd)", "lang": "en"},

    {"subreddit": "GlobalTalk",
     "perspective": "Global South (Mixed)", "lang": "en"},
]


# ─────────────────────────────────────────────
# 3. TRANSLATION
# ─────────────────────────────────────────────
def translate_text(text, source_lang):
    if source_lang == 'en' or not text:
        return text
    try:
        text = str(text)[:800]
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception:
        return text


# ─────────────────────────────────────────────
# 4. ARTICLE BODY SCRAPER (for link posts)
# ─────────────────────────────────────────────
def scrape_article_body(url: str, max_chars: int = 1000) -> str:
    """
    Reddit link posts point to external articles.
    This scrapes the actual article body from that link.
    Falls back silently on paywalls or blocks.
    """
    if not url or "reddit.com" in url:
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

        # Strip junk
        for tag in soup(["script", "style", "nav", "footer",
                         "header", "aside", "form", "figure",
                         "figcaption", "iframe", "noscript"]):
            tag.decompose()

        # Prefer <article> tag — cleaner text on most news sites
        article_tag = soup.find("article")
        paragraphs  = article_tag.find_all("p") if article_tag else soup.find_all("p")
        body        = " ".join(p.get_text(strip=True) for p in paragraphs)
        body        = " ".join(body.split())
        return body[:max_chars]

    except Exception:
        return ""


# ─────────────────────────────────────────────
# 5. COMMENT HARVESTER
# ─────────────────────────────────────────────
def get_top_comments(post, max_comments: int = 5, max_chars_each: int = 200) -> str:
    """
    Grabs the top N comments from a Reddit post.
    Comments reveal HOW the crowd is reacting to / framing the story —
    this is where propaganda narratives actually spread.

    Returns a single joined string for the AI to analyze.
    """
    try:
        post.comments.replace_more(limit=0)  # Don't fetch nested "load more"
        top_comments = []

        for comment in post.comments[:max_comments]:
            if not hasattr(comment, 'body'):
                continue
            body = comment.body.strip()

            # Skip deleted/removed/bot comments
            if body in ['[deleted]', '[removed]', '']:
                continue
            if len(body) < 10:  # Skip one-word reactions
                continue

            top_comments.append(body[:max_chars_each])

        return " | ".join(top_comments)

    except Exception:
        return ""


# ─────────────────────────────────────────────
# 6. MAIN FETCH FUNCTION
# ─────────────────────────────────────────────
def fetch_reddit_data():
    all_posts = []
    print("Starting the Reddit OSINT Siphon (Full Content Mode)...")

    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        reddit.user.me()
        print("Reddit API connected.\n")
    except Exception as e:
        print(f"Failed to connect to Reddit API. Check your .env file! Error: {e}")
        return []

    for target in TARGET_SUBS:
        print(f" -> r/{target['subreddit']} [{target['perspective']}]")
        try:
            subreddit = reddit.subreddit(target['subreddit'])
            count     = 0

            for post in subreddit.hot(limit=12):
                if post.stickied:
                    continue

                raw_title     = post.title.strip()
                english_title = translate_text(raw_title, target['lang'])

                # ── Self-text (for text posts like r/geopolitics analysis) ──
                selftext = ""
                if post.selftext and post.selftext not in ['', '[deleted]', '[removed]']:
                    selftext = translate_text(post.selftext[:800], target['lang'])

                # ── External article body (for link posts) ──
                external_url  = post.url if not post.is_self else ""
                article_body  = scrape_article_body(external_url) if external_url else ""

                # ── Top comments (crowd reaction / narrative spread) ──
                print(f"    Fetching comments: {raw_title[:55]}...")
                comments_text = get_top_comments(post)

                # ── Build full_text ──
                # Priority: title + selftext/article body + crowd comments
                # This gives the AI: WHAT happened + HOW it's framed + HOW crowd reacts
                parts = [p for p in [
                    english_title,
                    selftext,
                    article_body,
                    comments_text
                ] if p]
                full_text = " ".join(parts)[:2000]  # Cap at 2000 chars

                article = {
                    "source":            f"Reddit: r/{target['subreddit']}",
                    "perspective":       target['perspective'],
                    "original_language": target['lang'],
                    "raw_title":         raw_title,
                    "title":             english_title,
                    "full_text":         full_text,        # ← AI analyzes THIS
                    "published_date":    datetime.fromtimestamp(
                                            post.created_utc
                                        ).strftime("%Y-%m-%d %H:%M:%S"),
                    "link":              f"https://www.reddit.com{post.permalink}",
                    "fetch_timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_posts.append(article)
                count += 1

            print(f"    Done. {count} posts collected.\n")

        except Exception as e:
            print(f"  Failed to scrape r/{target['subreddit']}: {e}\n")

    return all_posts


# ─────────────────────────────────────────────
# 7. SAVE
# ─────────────────────────────────────────────
def save_to_csv(posts_list):
    if not posts_list:
        print("No data collected.")
        return

    df = pd.DataFrame(posts_list)
    os.makedirs("data/raw", exist_ok=True)
    filename = "data/raw/osint_reddit_latest.csv"
    df.to_csv(filename, index=False)

    print(f"Success! Saved {len(df)} Reddit posts to {filename}")
    print(f"Subreddits collected: {df['source'].value_counts().to_dict()}")

    print("\n--- Full Text Preview (first 3 posts) ---")
    for _, row in df.head(3).iterrows():
        print(f"\nSource:    {row['source']}")
        print(f"Title:     {row['title'][:80]}")
        print(f"Full Text: {row['full_text'][:200]}...")


if __name__ == "__main__":
    scraped_data = fetch_reddit_data()
    save_to_csv(scraped_data)