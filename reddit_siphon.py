import os
import pandas as pd
import praw
from datetime import datetime
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

# 1. Load the secret vault
load_dotenv()
CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
CLIENT_SECRET = os.getenv('REDDIT_SECRET')
USER_AGENT = os.getenv('REDDIT_USER_AGENT')

# 2. Define the Target Subreddits
TARGET_SUBS = [
    {"subreddit": "worldnews", "perspective": "Global (Mainstream)", "lang": "en"},
    {"subreddit": "geopolitics", "perspective": "Global (Analytical)", "lang": "en"},
    {"subreddit": "ukraine", "perspective": "Ukraine (Regional)", "lang": "en"},
    {"subreddit": "AskMiddleEast", "perspective": "Middle East (Crowd)", "lang": "en"}
]


def translate_text(text, source_lang):
    if source_lang == 'en' or not text:
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception:
        return text


def fetch_reddit_data():
    all_posts = []
    print("Starting the Reddit OSINT Siphon...")

    # Initialize the Reddit API client
    try:
        reddit = praw.Reddit(
            client_id=CLIENT_ID,
            client_secret=CLIENT_SECRET,
            user_agent=USER_AGENT
        )
        # Quick test to make sure it connected
        reddit.user.me()
    except Exception as e:
        print(f"Failed to connect to Reddit API. Check your .env file! Error: {e}")
        return []

    for target in TARGET_SUBS:
        print(f" -> Siphoning 'Hot' posts from r/{target['subreddit']}...")
        try:
            # Fetch the top 10 currently "Hot" posts from the subreddit
            subreddit = reddit.subreddit(target['subreddit'])
            for post in subreddit.hot(limit=10):

                # Skip pinned posts (usually rules or megathreads)
                if post.stickied:
                    continue

                raw_title = post.title
                english_title = translate_text(raw_title, target['lang'])

                article = {
                    "source": f"Reddit: r/{target['subreddit']}",
                    "perspective": target['perspective'],
                    "original_language": target['lang'],
                    "raw_title": raw_title,
                    "title": english_title,  # Our AI will analyze the title!
                    "published_date": datetime.fromtimestamp(post.created_utc).strftime("%Y-%m-%d %H:%M:%S"),
                    "link": f"https://www.reddit.com{post.permalink}",
                    "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_posts.append(article)

        except Exception as e:
            print(f"Failed to scrape r/{target['subreddit']}: {e}")

    return all_posts


def save_to_csv(posts_list):
    if not posts_list:
        print("No data collected.")
        return

    df = pd.DataFrame(posts_list)
    today_str = datetime.now().strftime("%Y_%m_%d")

    # Save it to the raw folder so our AI engine can pick it up!
    filename = "data/raw/osint_reddit_latest.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(filename, index=False)

    print(f"\nSuccess! Saved {len(df)} Reddit posts to {filename}")
    print("\n--- Reddit OSINT Preview ---")
    print(df[['source', 'title']].head(5))


if __name__ == "__main__":
    scraped_data = fetch_reddit_data()
    save_to_csv(scraped_data)