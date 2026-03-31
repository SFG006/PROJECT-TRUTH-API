import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict
from umap import UMAP
# Embeddings & Clustering
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ChromaDB
import chromadb
from chromadb.config import Settings

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE = "data/processed/master_tactics_latest.csv"
OUTPUT_REPORT = "data/processed/narrative_report.json"
CHROMA_DB_PATH = "data/chroma_store"
COLLECTION_NAME = "truth_tide_events"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DBSCAN_EPS = 0.40
DBSCAN_MIN_SAMPLES = 2

TACTIC_SEVERITY = {
    "Disinformation": 1.0,
    "Justification of Violence": 0.95,
    "Fear-Mongering / Alarmism": 0.90,
    "Othering / Scapegoating": 0.88,
    "Victim-Blaming": 0.85,
    "Emotional Manipulation": 0.72,
    "Ridicule / Delegitimization": 0.68,
    "Selective Framing / Bias": 0.60,
    "Nationalism / Pride": 0.55,
    "Moral / Religious Framing": 0.50,
    "Anti-Establishment": 0.45,
    "False Authority / Unverified Claims": 0.40,
    "Neutral Reporting": 0.05,
    "ERROR": 0.0,
}


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA (Fallback if not passed in RAM)
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    print(f"[1/5] Loading master data from: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Cannot find {filepath}. Run tactic_engine.py first.")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["title", "tactic_label"])
    df = df.reset_index(drop=True)
    print(f"      Loaded {len(df)} headlines.")
    return df


# ─────────────────────────────────────────────
# STEP 2 — EMBED + CLUSTER INTO EVENTS
# ─────────────────────────────────────────────
def embed_and_cluster(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    print(f"\n[2/5] Embedding {len(df)} headlines with '{EMBEDDING_MODEL}' for 3D Mapping...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    text_to_embed = df["full_text"].fillna(df["title"]).tolist()
    embeddings = model.encode(text_to_embed, show_progress_bar=True)

    print(f"      Running DBSCAN clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    cosine_dist_matrix = 1 - cosine_similarity(embeddings)
    cosine_dist_matrix = np.clip(cosine_dist_matrix, 0, 2)

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(cosine_dist_matrix)

    n_events = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)

    print(f"      Found {n_events} event clusters | {n_noise} standalone headlines.")
    return embeddings, labels


def generate_3d_coordinates(df: pd.DataFrame, embeddings: np.ndarray):
    print(f"\n[2.5/5] Generating 3D Semantic Map Coordinates (UMAP)...")
    reducer = UMAP(
        n_components=3,
        n_neighbors=10,
        min_dist=0.3,
        random_state=42
    )

    # Calculate the 3D positions based on the semantic embeddings
    coords = reducer.fit_transform(embeddings)

    # Attach them to the dataframe so they end up in the JSON
    df['x'] = coords[:, 0]
    df['y'] = coords[:, 1]
    df['z'] = coords[:, 2]

    print("      UMAP done. Appended X, Y, Z coordinates to data.")
    return df


# ─────────────────────────────────────────────
# STEP 3 — BUILD EVENT OBJECTS + NARRATIVE DELTA
# ─────────────────────────────────────────────
def compute_narrative_delta(perspectives: list[str], tactics: list[str]) -> dict:
    if not tactics:
        return {}

    framing_map = defaultdict(list)
    for p, t in zip(perspectives, tactics):
        framing_map[p].append(t)

    from collections import Counter
    tactic_counts = Counter(tactics)
    dominant = tactic_counts.most_common(1)[0][0]
    dominant_count = tactic_counts.most_common(1)[0][1]

    agreement_score = round(dominant_count / len(tactics), 3)
    severity_vals = [TACTIC_SEVERITY.get(t, 0.0) for t in tactics]
    severity_score = round(float(np.mean(severity_vals)), 3)

    unique_tactics = list(set(tactics))

    return {
        "dominant_tactic": dominant,
        "agreement_score": agreement_score,
        "delta_score": round(1 - agreement_score, 3),
        "severity_score": severity_score,
        "unique_tactics": unique_tactics,
        "framing_map": {k: list(set(v)) for k, v in framing_map.items()},
    }


def build_event_clusters(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    print(f"\n[3/5] Building event objects and computing Narrative Deltas...")
    events = []
    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        cluster_df = df[mask].copy()

        if label == -1:
            for idx, row in cluster_df.iterrows():
                event = {
                    "event_id": f"standalone_{idx}",
                    "is_event_cluster": False,
                    "headline_count": 1,
                    "representative_title": row["title"],
                    "sources": [row["source"]],
                    "perspectives": [row.get("perspective", "Unknown")],
                    "headlines": [row.to_dict()],
                    "narrative_delta": compute_narrative_delta([row.get("perspective", "Unknown")],
                                                               [row["tactic_label"]]),
                    "timestamp": datetime.now().isoformat(),
                }
                events.append(event)
            continue

        # Ensure the coordinates are explicitly kept in the dictionary
        headlines = cluster_df.to_dict(orient="records")

        # Clean up any NaN values that might break the JSON parser
        for h in headlines:
             if pd.isna(h.get('x')): h['x'] = 0.0
             if pd.isna(h.get('y')): h['y'] = 0.0
             if pd.isna(h.get('z')): h['z'] = 0.0
        perspectives = cluster_df["perspective"].tolist()
        tactics = cluster_df["tactic_label"].tolist()
        sources = cluster_df["source"].tolist()

        # Grab the first headline as representative
        representative = headlines[0]["title"]

        narrative = compute_narrative_delta(perspectives, tactics)
        event = {
            "event_id": f"evt_{label}",
            "is_event_cluster": True,
            "headline_count": int(mask.sum()),
            "representative_title": representative,
            "sources": list(set(sources)),
            "perspectives": list(set(perspectives)),
            "headlines": headlines,
            "narrative_delta": narrative,
            "timestamp": datetime.now().isoformat(),
        }
        events.append(event)

    events.sort(key=lambda e: e["narrative_delta"].get("delta_score", 0), reverse=True)
    print(f"      Built {len(events)} event objects.")
    return events


# ─────────────────────────────────────────────
# STEP 4 — PERSIST TO CHROMADB
# ─────────────────────────────────────────────
def persist_to_chromadb(df: pd.DataFrame, embeddings: np.ndarray):
    print(f"\n[4/5] Persisting {len(df)} vectors to ChromaDB at '{CHROMA_DB_PATH}'...")
    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    def make_id(row):
        raw = f"{row['source']}_{row.get('published_date', '?')}_{row['title'][:40]}"
        return raw.replace(" ", "_").replace("/", "-")[:100]

    ids = [make_id(row) for _, row in df.iterrows()]
    embeddings_list = embeddings.tolist()

    metadatas = [
        {
            "source": str(row.get("source", "")),
            "perspective": str(row.get("perspective", "")),
            "tactic_label": str(row.get("tactic_label", "")),
            "tactic_confidence": float(row.get("tactic_confidence", 0.0)),
            "published_date": str(row.get("published_date", "")),
            "link": str(row.get("link", "")),
            "full_text": str(row.get("full_text", ""))[:500],
        }
        for _, row in df.iterrows()
    ]
    documents = df["title"].tolist()

    BATCH = 50
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i + BATCH],
            embeddings=embeddings_list[i:i + BATCH],
            documents=documents[i:i + BATCH],
            metadatas=metadatas[i:i + BATCH],
        )
    total = collection.count()
    print(f"      ChromaDB now holds {total} total vectors in '{COLLECTION_NAME}'.")


# ─────────────────────────────────────────────
# STEP 5 — EXPORT NARRATIVE REPORT
# ─────────────────────────────────────────────
def build_summary_stats(df: pd.DataFrame, events: list[dict]) -> dict:
    tactic_dist = df["tactic_label"].value_counts().to_dict()
    source_dist = df["source"].value_counts().to_dict()
    event_clusters = [e for e in events if e["is_event_cluster"]]
    contested_events = [e for e in event_clusters if e["narrative_delta"].get("delta_score", 0) >= 0.5]

    return {
        "total_headlines": int(len(df)),
        "total_sources": int(df["source"].nunique()),
        "total_tactics": int(df["tactic_label"].nunique()),
        "total_event_clusters": len(event_clusters),
        "contested_events": len(contested_events),
        "avg_severity": round(
            float(np.mean([TACTIC_SEVERITY.get(t, 0) for t in df["tactic_label"]])), 3
        ),
        "tactic_distribution": tactic_dist,
        "source_distribution": source_dist,
        "run_timestamp": datetime.now().isoformat(),
    }


def export_report(events: list[dict], summary: dict, output_path: str):
    print(f"\n[5/5] Exporting Narrative Report to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report = {
        "meta": summary,
        "events": events,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"      Report saved. Size: {size_kb:.1f} KB")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
# Notice how this now accepts preloaded_df directly from tactic_engine.py
def run_semantic_engine(preloaded_df=None):
    print("=" * 56)
    print("  TRUTH-TIDE · Semantic Engine v2 (Groq Pipeline)")
    print("=" * 56)

    # If tactic_engine passed the data in RAM, use it. Otherwise, load from CSV.
    df = preloaded_df if preloaded_df is not None else load_data(INPUT_FILE)

    embeddings, labels = embed_and_cluster(df)
    df = generate_3d_coordinates(df, embeddings)

    # Save the dataframe back to the CSV so the API can read the X, Y, Z columns
    df.to_csv(INPUT_FILE, index=False)
    # ---------------------

    events = build_event_clusters(df, labels)

    persist_to_chromadb(df, embeddings)
    summary = build_summary_stats(df, events)
    export_report(events, summary, OUTPUT_REPORT)

    print("\n" + "─" * 56)
    print("  NARRATIVE INTELLIGENCE SUMMARY")
    print("─" * 56)
    print(f"  Headlines analysed : {summary['total_headlines']}")
    print(f"  Event clusters     : {summary['total_event_clusters']}")
    print(f"  Contested events   : {summary['contested_events']}  (delta ≥ 0.50)")
    print(f"  Avg. severity      : {summary['avg_severity']}")
    print()
    print("  TOP 5 MOST CONTESTED EVENTS:")
    for e in [x for x in events if x["is_event_cluster"]][:5]:
        nd = e["narrative_delta"]
        print(f"  [{nd.get('delta_score', '?'):.2f} Δ] {e['representative_title'][:65]}...")
        print(f"        Tactics seen: {', '.join(nd.get('unique_tactics', []))}")
    print("─" * 56)
    print("  Done. JSON Report is ready for the dashboard!\n")


if __name__ == "__main__":
    run_semantic_engine()