import os
import pandas as pd
from datetime import datetime
from tweety import Twitter
from deep_translator import GoogleTranslator

# 1. Define the OSINT Targets on X
TARGET_ACCOUNTS = [
    {"username": "mfa_russia", "perspective": "Russia (State)", "lang": "en"},  # Russian Ministry of Foreign Affairs
    {"username": "ZelenskyyUa", "perspective": "Ukraine (State)", "lang": "en"},
    {"username": "AJA_Egypt", "perspective": "Middle East (Raw)", "lang": "ar"}
]


def translate_text(text, source_lang):
    if source_lang == 'en' or not text:
        return text
    try:
        return GoogleTranslator(source=source_lang, target='en').translate(text)
    except Exception:
        return text


def fetch_x_data():
    all_tweets = []
    print("Starting the X (Twitter) OSINT Siphon...")

    # Initialize the unofficial Twitter scraper
    app = Twitter("session")

    for account in TARGET_ACCOUNTS:
        print(f" -> Siphoning raw tweets from @{account['username']}...")
        try:
            # Fetch the latest tweets from the user
            tweets = app.get_tweets(account['username'], pages=1)

            # Grab the top 10 tweets
            for tweet in tweets[:10]:
                raw_text = tweet.text.replace('\n', ' ')

                # Skip retweets or empty text to keep our data clean
                if not raw_text or tweet.is_retweet:
                    continue

                english_text = translate_text(raw_text, account['lang'])

                article = {
                    "source": f"X: @{account['username']}",
                    "perspective": account['perspective'],
                    "original_language": account['lang'],
                    "raw_title": raw_text,
                    "title": english_text,  # Our AI will analyze this!
                    "published_date": str(tweet.created_on),
                    "link": f"https://x.com/{account['username']}/status/{tweet.id}",
                    "fetch_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                all_tweets.append(article)

        except Exception as e:
            print(f"Failed to scrape @{account['username']}. X might be blocking us: {e}")

    return all_tweets


def save_to_csv(tweets_list):
    if not tweets_list:
        print("No data collected. X might have blocked the scraper.")
        return

    df = pd.DataFrame(tweets_list)
    today_str = datetime.now().strftime("%Y_%m_%d")

    # Save it to the raw folder so our AI engine can pick it up!
    filename = f"data/raw/osint_x_{today_str}.csv"
    os.makedirs("data/raw", exist_ok=True)
    df.to_csv(filename, index=False)

    print(f"\nSuccess! Saved {len(df)} Tweets to {filename}")
    print("\n--- X OSINT Preview ---")
    print(df[['source', 'title']].head(5))


if __name__ == "__main__":
    scraped_data = fetch_x_data()
    save_to_csv(scraped_data)