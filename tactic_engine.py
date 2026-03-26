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
    # --- Emotional Manipulation ---
    "uses words like catastrophe, crisis, or disaster to provoke fear":
        "Fear-Mongering",

    "tells a personal tragedy story to make the reader feel guilt or pity":
        "Emotional Manipulation",

    "warns of imminent collapse, invasion, or irreversible damage":
        "Alarmism",

    # --- Identity & Group Dynamics ---
    "celebrates a nation, army, or people as heroic, superior, or victorious":
        "Nationalism / Pride",

    "portrays a religion, ethnicity, or political group as dangerous or criminal":
        "Othering / Scapegoating",

    "attacks governments, media, or institutions as corrupt or untrustworthy":
        "Anti-Establishment",

    "uses God, religion, scripture, or moral duty to justify a political action":
        "Moral / Religious Framing",

    # --- Narrative Distortion ---
    "describes civilians or protesters killed as responsible for their own deaths":
        "Victim-Blaming",

    "describes a military strike, killing, or war as justified, necessary, or heroic":
        "Justification of Violence",

    "contradicts known facts or presents events that did not happen as real":
        "Disinformation / Fabrication",

    "highlights only facts that support one side while ignoring contradictory evidence":
        "Selective Framing / Bias",

    "mocks, ridicules, or dismisses a leader, country, or group as stupid or weak":
        "Ridicule / Delegitimization",

    # --- Authority & Credibility ---
    "quotes an unnamed official, anonymous source, or unnamed expert as proof":
        "False Authority",

    "reports unconfirmed battlefield claims, casualty numbers, or territorial gains":
        "Unverified Claims",

    # --- Neutral ---
    "reports a confirmed fact, official statement, or verified event without bias":
        "Neutral Reporting",
}

candidate_labels = list(TACTIC_DESCRIPTIONS.keys())


def get_tactic(text):
    try:
        # Use full_text — cap at 512 tokens (model limit)
        # Take first 512 chars — intro sentences carry the most framing signal
        result = classifier(
            str(text)[:512],
            candidate_labels=candidate_labels,
            hypothesis_template="This text {}.",
            multi_label=True
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
    n_components=2,
    n_neighbors=10,    # lower = tighter local clusters
    min_dist=0.3,      # higher = more visual spread
    random_state=42    # fixed seed = same layout every run
)
reduced_embeddings = reducer.fit_transform(embeddings)
df['x'] = reduced_embeddings[:, 0]
df['y'] = reduced_embeddings[:, 1]

print(f"UMAP done. X range: {df['x'].min():.2f} → {df['x'].max():.2f}")
print(f"           Y range: {df['y'].min():.2f} → {df['y'].max():.2f}")

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
run_semantic_engine()