import os
import json
import time
import pandas as pd
import glob
from tqdm import tqdm
from groq import Groq

# ─────────────────────────────────────────────
# STEP 1 — Load and Combine Data
# ─────────────────────────────────────────────
raw_files = glob.glob("data/raw/*.csv")
if not raw_files:
    print("No raw data found! Drop a CSV in data/raw/ first.")
    exit()

print(f"Found {len(raw_files)} raw data files. Combining...")
df_list  = [pd.read_csv(file) for file in raw_files]
df = pd.concat(df_list, ignore_index=True).drop_duplicates(subset=['title', 'source'])
df = df[~df['title'].astype(str).str.contains('<<<<<<|>>>>>>|=======', regex=True)].reset_index(drop=True)

if 'full_text' not in df.columns:
    df['full_text'] = df['title']
else:
    df['full_text'] = df['full_text'].fillna(df['title'])
    df.loc[df['full_text'] == '', 'full_text'] = df['title']

print(f"Total unique articles: {len(df)}")

df = df.head(400) #reducing articles to 400

# ─────────────────────────────────────────────
# STEP 2 — Groq LLM Tactic Classification
# ─────────────────────────────────────────────
print("\nWaking up the Groq Llama-3 Agent...")

# Automatically picks up the GROQ_API_KEY environment variable
client = Groq()

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


def get_tactic_llm(text):
    prompt = f"""
    You are an expert media analyst. Analyze this news article's framing.

    Step 1: Assign it a broad category from this exact list:
    {TACTIC_DEFINITIONS}

    Step 2: Invent a highly specific, 2-to-4 word name for the exact micro-tactic being used (e.g., "Nuclear Threat Exaggeration", "Historical Revisionism", "False Equivalence").

    Article Text:
    "{str(text)}"

    Respond ONLY with a valid JSON object matching this exact structure:
    {{
        "tactic_label": "Exact Name of broad category from the list",
        "specific_technique": "The specific 2-to-4 word tactic you identified",
        "confidence": "Float between 0.0 and 1.0 representing your certainty",
        "reasoning": "A one-sentence explanation of why this tactic applies."
    }}
    """
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="openai/gpt-oss-20b",
            temperature=0,
            response_format={"type": "json_object"}
        )

        result = json.loads(chat_completion.choices[0].message.content)
        return (
            result.get('tactic_label', 'Neutral Reporting'),
            result.get('specific_technique', 'None'),  # Grab the new invented tactic
            float(result.get('confidence', 0.0)),
            result.get('reasoning', '')
        )
    except Exception as e:
        return "ERROR", "ERROR", 0.0, str(e)


results = []
for count, (index, row) in enumerate(tqdm(df.iterrows(), total=len(df), desc="Groq API Processing"), 1):
    tactic, specific, conf, reasoning = get_tactic_llm(row['full_text'])
    results.append({
        'tactic_label': tactic,
        'specific_technique': specific,
        'tactic_confidence': conf,
        'llm_reasoning': reasoning
    })

    if count % 20 == 0 and count != len(df):
        print(f"\nProcessed {count} articles. Sleeping for 60 seconds to reset token limits...")
        time.sleep(60)

# Reattach the results to the original DataFrame
results_df = pd.DataFrame(results)
df = pd.concat([df.reset_index(drop=True), results_df.reset_index(drop=True)], axis=1)
# ─────────────────────────────────────────────
# STEP 3 — Auto-trigger Semantic Engine (In-Memory Handoff)
# ─────────────────────────────────────────────
print("\nHanding off to Semantic Engine...")
from semantic_engine import run_semantic_engine

# We only pass the dataframe. The semantic engine will handle embedding the text for clustering.
run_semantic_engine(preloaded_df=df)