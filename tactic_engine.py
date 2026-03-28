# tactic_engine.py
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from umap import UMAP
import os
import glob
from tqdm import tqdm
from datetime import datetime

# ─────────────────────────────────────────────
# STEP 1 — Load and Combine ALL Raw Data
# ─────────────────────────────────────────────
raw_files = glob.glob("data/raw/*.csv")
if not raw_files:
    print("No raw data found! Run data_siphon.py and reddit_siphon.py first.")
    exit()

print(f"Found {len(raw_files)} raw data files. Combining them...")
df_list = [pd.read_csv(file) for file in raw_files]
df = pd.concat(df_list, ignore_index=True)
df = df.drop_duplicates(subset=['title', 'source'])

# ── Drop any git conflict rows that snuck into the CSV ──
df = df[~df['title'].astype(str).str.contains('<<<<<<|>>>>>>|=======', regex=True)]
df = df.reset_index(drop=True)

# ── If full_text column is missing (e.g. old Reddit CSV), fall back to title ──
# This keeps old reddit_siphon.py data compatible
if 'full_text' not in df.columns:
    df['full_text'] = df['title']
else:
    # Fill any actual NaNs first
    df['full_text'] = df['full_text'].fillna(df['title'])
    # Pinpoint empty strings and safely inject the title
    df.loc[df['full_text'] == '', 'full_text'] = df['title']

print(f"Total unique articles to analyze: {len(df)}")
full_text_count = df[df['full_text'] != df['title']].shape[0]
print(f"Articles with full body text: {full_text_count} / {len(df)}")

# ─────────────────────────────────────────────
# STEP 2 — Zero-Shot Tactic Classification
# ─────────────────────────────────────────────
print("\nWaking up the Zero-Shot Tactics Engine...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# ── Rewritten descriptions — specific language patterns, not abstract concepts ──
# These match actual sentence patterns found in news bodies, not just headlines
TACTIC_DESCRIPTIONS = {
    # --- Emotional Manipulation (Expanded Triggers) ---
    "provokes panic by warning of an imminent catastrophe or irreversible damage": "Fear-Mongering / Alarmism",
    "exaggerates a threat to make the reader feel terrified or anxious": "Fear-Mongering / Alarmism",

    "tells a tragic personal story to manipulate the reader's sympathy or guilt": "Emotional Manipulation",
    "uses highly charged emotional language to bypass logical reasoning": "Emotional Manipulation",

    # --- Identity & Group Dynamics ---
    "glorifies a specific nation, military, or group as morally superior and heroic": "Nationalism / Pride",

    "dehumanizes a specific religion, ethnicity, or group by framing them as dangerous": "Othering / Scapegoating",
    "blames a minority or opposing group for a complex systemic problem": "Othering / Scapegoating",

    "frames established institutions, media, or governments as inherently corrupt": "Anti-Establishment",

    "claims a political or military action is mandated by God or moral righteousness": "Moral / Religious Framing",

    # --- Narrative Distortion ---
    "claims that victims of violence or oppression are responsible for their own suffering": "Victim-Blaming",

    "frames a military strike, assassination, or act of war as a necessary and heroic act": "Justification of Violence",

    "presents a completely fabricated event or directly contradicts proven historical facts": "Disinformation",

    "presents a heavily biased, one-sided narrative while deliberately omitting opposing facts": "Selective Framing / Bias",

    "uses sarcasm and insults to portray a political leader or nation as weak and incompetent": "Ridicule / Delegitimization",

    # --- Authority & Credibility ---
    "relies on anonymous sources or unnamed officials to make a severe allegation": "False Authority / Unverified Claims",
    "reports unconfirmed casualty numbers or battlefield victories as absolute facts": "False Authority / Unverified Claims",

    # --- Neutral ---
    "reports verified events, data, and official statements using objective, balanced language": "Neutral Reporting",
}
candidate_labels = list(TACTIC_DESCRIPTIONS.keys())


def get_tactic(text):
    try:
        # Use full_text — cap at 512 tokens (model limit)
        # Take first 512 chars — intro sentences carry the most framing signal
        result = classifier(
            str(text),
            candidate_labels=candidate_labels,
            hypothesis_template="This text {}.",
            multi_label=True,
            truncation=True
        )
        top_description = result['labels'][0]
        confidence      = round(result['scores'][0], 4)

        # Tier 1: High confidence — trust the label
        if confidence >= 0.75:
            return TACTIC_DESCRIPTIONS[top_description], confidence

        # Tier 2: Medium confidence — flag as uncertain
        elif confidence >= 0.55:
            tactic = TACTIC_DESCRIPTIONS[top_description]
            return f"Uncertain / {tactic}", confidence

        # Tier 3: Low confidence — genuinely ambiguous
        else:
            return "Neutral Reporting", confidence

    except Exception as e:
        print(f"Error on text: {e}")
        return "ERROR", 0.0


print("\nAnalyzing articles for psychological tactics (using full text)...")
# Initialize the progress bar for Pandas
tqdm.pandas(desc="AI Tactic Analysis")

# Use progress_apply instead of apply
df[['tactic_label', 'tactic_confidence']] = df['full_text'].progress_apply(
    lambda x: pd.Series(get_tactic(x))
)

# ─────────────────────────────────────────────
# STEP 3 — Semantic Map Coordinates (UMAP)
# ─────────────────────────────────────────────
print("\nCalculating semantic topography (UMAP)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed full_text for richer semantic positioning on the map
embeddings = embedding_model.encode(
    df['full_text'].tolist(),
    show_progress_bar=True
)

reducer = UMAP(
    n_components=3,
    n_neighbors=10,    # lower = tighter local clusters
    min_dist=0.3,      # higher = more visual spread
    random_state=42    # fixed seed = same layout every run
)
reduced_embeddings = reducer.fit_transform(embeddings)
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]
df['z'] = reduced_embeddings[:, 2]

print(f"UMAP done. X range: {df['x'].min():.2f} → {df['x'].max():.2f}")
print(f"           Y range: {df['y'].min():.2f} → {df['y'].max():.2f}")
print(f"           Z range: {df['z'].min():.2f} → {df['z'].max():.2f}")

# ─────────────────────────────────────────────
# STEP 4 — Save Master File
# ─────────────────────────────────────────────
os.makedirs("data/processed", exist_ok=True)
output_filename = "data/processed/master_tactics_latest.csv"
df.to_csv(output_filename, index=False)
print(f"\nSuccess! Master data saved to: {output_filename}")

print("\n--- Tactic Distribution ---")
print(df['tactic_label'].value_counts().to_string())

print("\n--- Sample Preview ---")
print(df[['source', 'title', 'tactic_label', 'tactic_confidence']].head(10).to_string())

# ─────────────────────────────────────────────
# STEP 5 — Auto-trigger Semantic Engine
# ─────────────────────────────────────────────
print("\nHanding off to Semantic Engine...")
from semantic_engine import run_semantic_engine

# Pass your existing dataframe and the embeddings you made for UMAP directly!
run_semantic_engine(preloaded_df=df, precomputed_embeddings=embeddings)