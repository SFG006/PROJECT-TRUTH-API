import os
import json
import time
import pandas as pd
import glob
from tqdm import tqdm
from groq import Groq
import itertools
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# STEP 1 — Load and Combine Data
# ─────────────────────────────────────────────
# Search the data/raw directory for any CSV files
raw_files = glob.glob("data/raw/*.csv")
if not raw_files:
    print("No raw data found! Drop a CSV in data/raw/ first.")
    exit()

print(f"Found {len(raw_files)} raw data files. Combining...")
# Read all found CSVs into a list of pandas DataFrames
df_list = [pd.read_csv(file) for file in raw_files]
# Combine the list into a single master DataFrame and drop exact duplicate articles
df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['title', 'source'])
# Filter out any messy rows containing Git conflict markers (<<<<<<, >>>>>>, =======)
df = df[~df['title'].astype(str).str.contains('<<<<<<|>>>>>>|=======', regex=True)].reset_index(drop=True)

# Ensure there is a 'full_text' column for the LLM to read; if missing or empty, use the 'title'
if 'full_text' not in df.columns:
    df['full_text'] = df['title']
else:
    df['full_text'] = df['full_text'].fillna(df['title'])
    df.loc[df['full_text'] == '', 'full_text'] = df['title']

print(f"Total unique articles: {len(df)}")

# ─────────────────────────────────────────────
# STEP 2 — Groq LLM Tactic Classification
# ─────────────────────────────────────────────
print("\nWaking up the Groq Agents...")

# Fetch both API keys from the local environment
key1 = os.environ.get("GROQ_API_KEY_1")
key2 = os.environ.get("GROQ_API_KEY_2")

active_clients = []
if key1:
    active_clients.append(Groq(api_key=key1))
if key2:
    active_clients.append(Groq(api_key=key2))

# delay configuration to avoid the 8,000 TPM limit
if len(active_clients) == 2:
    print("Success: 2 API keys detected! Running at double speed (30s delay).")
    DELAY_TIME = 30
elif len(active_clients) == 1:
    print("Warning: Only 1 API key detected. Running at safe speed (60s delay) to prevent crashes.")
    DELAY_TIME = 60
else:
    print("Warning: No custom keys detected. Falling back to default GROQ_API_KEY (60s delay).")
    active_clients = [Groq()]
    DELAY_TIME = 60

# The strict catalog of media tactics the LLM is allowed to choose from
TACTIC_DEFINITIONS = """
- "Fear-Mongering / Alarmism": Exaggerating threats to provoke panic.
- "Emotional Manipulation": Using tragic stories or charged language to elicit sympathy or guilt.
- "Nationalism / Pride": Glorifying a nation or military as morally superior.
- "Othering / Scapegoating": Blaming a minority or specific group for systemic problems.
- "Anti-Establishment": Framing institutions or media as inherently corrupt.
- "Moral / Religious Framing": Claiming actions are mandated by God or moral righteousness.
- "Victim-Blaming": Asserting victims are responsible for their own suffering.
- "Justification of Violence": Framing military strikes or war as necessary and heroic.
- "Disinformation": Presenting completely fabricated events.
- "Selective Framing / Bias": Deliberately omitting opposing facts to create a one-sided narrative.
- "Ridicule / Delegitimization": Using mockery to portray leaders/nations as weak.
- "False Authority / Unverified Claims": Relying on anonymous sources for severe allegations.
- "Neutral Reporting": Objective, balanced, and emotionally detached reporting.
"""


def clean_json_string(raw_str):
    # Failsafe cleaner: strips markdown block wrappers (```json ... ```) if the LLM disobeys formatting rules
    cleaned = raw_str.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").replace("json", "", 1).strip()

    return cleaned


def get_tactic_llm_batch(batch_items, active_client):
    # Convert the python dictionaries of articles into a JSON string for the prompt payload
    batch_json = json.dumps(batch_items, indent=2)

    # prompt engineering block
    prompt = f"""
    You are an expert media analyst. Analyze the framing for this list of news articles.

    Step 1: Assign each article a broad category from this exact list:
    {TACTIC_DEFINITIONS}

    Step 2: Invent a highly specific, 2 to 4 word name for the exact micro-tactic being used.

    Below is a JSON array of articles. Each has an "id" and "text":
    {batch_json}

    CRITICAL JSON RULES:
    1. Respond ONLY with a valid JSON array of objects. Do not use markdown.
    2. Enclose all JSON keys and values in standard double quotes ("). 
    3. If you need to quote specific text from the article inside your reasoning, DO NOT use quotes. Use asterisks instead (e.g., The author uses the phrase *nuclear threat* to cause panic).

    Return ONLY raw JSON matching this exact structure:
    [
        {{
            "id": <exact integer id from the input>,
            "tactic_label": "Exact Name of broad category from the list",
            "specific_technique": "The specific 2 to 4 word tactic you identified",
            "confidence": <float between 0.0 and 1.0>,
            "reasoning": "A one-sentence explanation."
        }}
    ]
    """
    #Retry Loop to handle network drops and LLM formatting typos
    MAX_RETRIES = 3
    for attempt in range(MAX_RETRIES):
        try:
            # Dynamic temperature: increases slightly on each retry so the LLM doesn't repeat the exact same typo
            current_temp = 0.0 + (attempt * 0.2)

            # Fire the request to the Groq API using the actively assigned key
            chat_completion = active_client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model="openai/gpt-oss-20b",
                temperature=current_temp,
                max_tokens=2000
            )

            raw_content = chat_completion.choices[0].message.content

            # Protect against empty API responses
            if not raw_content:
                raise ValueError("API returned an empty string.")

            # Clean and parse the string into a Python list of dictionaries
            cleaned_content = clean_json_string(raw_content)

            return json.loads(cleaned_content), "SUCCESS"

        except json.JSONDecodeError as je:
            # Caught an LLM typo (e.g., unescaped quotes), trigger a retry            print(f"  -> Attempt {attempt + 1} Failed (JSON Error): {je}")
            time.sleep(3)
        except Exception as e:
            # Caught a network or rate limit error
            error_str = str(e).lower()
            print(f"  -> Attempt {attempt + 1} Failed (API Error): {e}")

            # Quota Kill Switch: immediately aborts if a key hits its 200k daily token limit
            if "rate limit" in error_str and ("tokens per day" in error_str or "tpd" in error_str):
                return None, "QUOTA_EXHAUSTED"

            time.sleep(3)
    # Return None if the batch failed 3 consecutive times
    print("  -> Batch completely failed after 3 attempts. Leaving blank.")
    return None, "FAILED"

# Initialize variables for the processing loop
BATCH_SIZE = 5
results_list = [None] * len(df)

# Iterate through the full DataFrame in chunks of 5 articles
for batch_index, i in enumerate(tqdm(range(0, len(df), BATCH_SIZE), desc="Groq API Processing")):

    # Alternating Key: even batches get key 1, odd batches get key 2
    if len(active_clients) == 2:
        current_client = active_clients[batch_index % 2]
    else:
        current_client = active_clients[0]

    # Slice the chunk from the dataframe and truncate articles to 1500 chars to save tokens
    batch_df = df.iloc[i:i + BATCH_SIZE]
    batch_items = [{"id": idx, "text": str(row['full_text'])[:1500]} for idx, row in batch_df.iterrows()]

    # Send the batch to the LLM function
    response_array, status = get_tactic_llm_batch(batch_items, current_client)

    # If the kill switch triggered, break the loop to save currently processed data
    if status == "QUOTA_EXHAUSTED":
        print("\nDaily API Quota exhausted! Halting loop and saving current progress...")
        break

    # Unpack the valid JSON payload and insert results directly into their correct index position
    if response_array and isinstance(response_array, list):
        for res in response_array:
            item_id = res.get('id')
            if item_id is not None and item_id < len(results_list):
                results_list[item_id] = {
                    "tactic_label": res.get("tactic_label", "Neutral Reporting"),
                    "specific_technique": res.get("specific_technique", "None"),
                    "tactic_confidence": float(res.get("confidence", 0.0)),
                    "llm_reasoning": res.get("reasoning", "")
                }
    # Rest the API keys to reset the 8,000 Tokens Per Minute limit
    time.sleep(DELAY_TIME)

# Failsafe: convert any unprocessed/failed rows (None) into empty dictionaries before converting to DataFrame
cleaned_results = [res if res is not None else {} for res in results_list]
res_df = pd.DataFrame(cleaned_results)

# Append the new classification columns to the original DataFrame
df = pd.concat([df, res_df], axis=1)

# Drop any rows that failed to process so the CSV is perfectly clean
df = df.dropna(subset=['tactic_label']).reset_index(drop=True)

print(f"\nSaving {len(df)} successfully processed articles to master CSV...")
df.to_csv("data/processed/master_tactics_latest.csv", index=False)

# ─────────────────────────────────────────────
# STEP 3  Auto trigger Semantic Engine (In Memory Handoff)
# ─────────────────────────────────────────────
print("\nHanding off to Semantic Engine...")
from semantic_engine import run_semantic_engine

# Pass the successfully enriched DataFrame down the pipeline
run_semantic_engine(preloaded_df=df)