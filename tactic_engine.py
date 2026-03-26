# tactic_engine.py
import pandas as pd
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from umap import UMAP
import os
import glob
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
print(f"Total unique headlines to analyze: {len(df)}")

# ─────────────────────────────────────────────
# STEP 2 — Zero-Shot Tactic Classification
# ─────────────────────────────────────────────
print("\nWaking up the Zero-Shot Tactics Engine...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

TACTIC_DESCRIPTIONS = {
    # --- Emotional Manipulation ---
    "exaggerates threats to create panic and fear":           "Fear-Mongering",
    "uses emotional stories to bypass rational thinking":     "Emotional Manipulation",
    "creates a sense of urgency or impending doom":           "Alarmism",

    # --- Identity & Group Dynamics ---
    "appeals to a sense of national superiority and loyalty": "Nationalism / Pride",
    "portrays one group as an enemy of society":              "Othering / Scapegoating",
    "promotes distrust of institutions or authorities":       "Anti-Establishment",
    "uses religious or moral framing to justify a position":  "Moral / Religious Framing",

    # --- Narrative Distortion ---
    "blames the victims of harm instead of the perpetrators":        "Victim-Blaming",
    "frames aggressive or violent acts as necessary and righteous":  "Justification of Violence",
    "presents a false or misleading version of events":              "Disinformation / Fabrication",
    "uses selective facts to push a one-sided narrative":            "Selective Framing / Bias",
    "uses mockery and ridicule to dismiss opposing views":           "Ridicule / Delegitimization",

    # --- Authority & Credibility ---
    "cites authority figures or experts to lend false credibility":  "False Authority",
    "uses vague or unverifiable claims to support a position":       "Unverified Claims",

    # --- Neutral ---
    "is objective, fact-based reporting without emotional manipulation": "Neutral Reporting",
}

candidate_labels = list(TACTIC_DESCRIPTIONS.keys())

def get_tactic(text):
    try:
        result = classifier(
            str(text)[:512],
            candidate_labels=candidate_labels,
            hypothesis_template="This text {}.",
            multi_label=True
        )
        top_description = result['labels'][0]
        confidence = round(result['scores'][0], 4)

        if confidence >= 0.70:
            return TACTIC_DESCRIPTIONS[top_description], confidence
        elif confidence >= 0.50:
            return f"Uncertain / {TACTIC_DESCRIPTIONS[top_description]}", confidence
        else:
            return "Neutral Reporting", confidence

    except Exception as e:
        print(f"Error on text: {e}")
        return "ERROR", 0.0

print("\nAnalyzing headlines for psychological tactics...")
df[['tactic_label', 'tactic_confidence']] = df['title'].apply(
    lambda x: pd.Series(get_tactic(x))
)

# ─────────────────────────────────────────────
# STEP 3 — Semantic Map Coordinates (UMAP)
# ─────────────────────────────────────────────
print("\nCalculating semantic topography (UMAP)...")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = embedding_model.encode(df['title'].tolist(), show_progress_bar=True)

# UMAP spreads clusters into actual 2D space — no more diagonal line
reducer = UMAP(
    n_components=2,
    n_neighbors=10,    # lower = tighter local clusters
    min_dist=0.3,      # higher = more visual spread between points
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
print("\n--- Tactics Preview ---")
print(df[['source', 'title', 'tactic_label']].head(10))

# ─────────────────────────────────────────────
# STEP 5 — Auto-trigger Semantic Engine
# ─────────────────────────────────────────────
print("\nHanding off to Semantic Engine...")
from semantic_engine import run_semantic_engine
run_semantic_engine()