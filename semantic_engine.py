# semantic_engine.py
# Truth-Tide: Semantic Intelligence Layer
# Responsibilities:
#   1. Load master_tactics_latest.csv
#   2. Cluster headlines into "Events" using DBSCAN on sentence embeddings
#   3. Compute Narrative Delta: how framings diverge across perspectives for the same event
#   4. Persist semantic fingerprints to ChromaDB for cross-session memory
#   5. Export a rich narrative_report.json for the dashboard

import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
from collections import defaultdict

# Embeddings
from sentence_transformers import SentenceTransformer

# Clustering
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity

# ChromaDB — persistent vector store
import chromadb
from chromadb.config import Settings

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
INPUT_FILE        = "data/processed/master_tactics_latest.csv"
OUTPUT_REPORT     = "data/processed/narrative_report.json"
CHROMA_DB_PATH    = "data/chroma_store"
COLLECTION_NAME   = "truth_tide_events"

EMBEDDING_MODEL   = "all-MiniLM-L6-v2"

# DBSCAN: how tight a cluster must be to be called one "event"
# eps=0.25 means headlines must be within 0.25 cosine distance
# min_samples=2 means at least 2 headlines to form an event cluster
DBSCAN_EPS        = 0.25
DBSCAN_MIN_SAMPLES = 2

# Tactic severity weights — used to score how "loaded" a cluster is
TACTIC_SEVERITY = {
    "Disinformation / Fabrication":  1.0,
    "Justification of Violence":     0.95,
    "Fear-Mongering":                0.90,
    "Othering / Scapegoating":       0.88,
    "Victim-Blaming":                0.85,
    "Alarmism":                      0.75,
    "Emotional Manipulation":        0.72,
    "Ridicule / Delegitimization":   0.68,
    "Selective Framing / Bias":      0.60,
    "Nationalism / Pride":           0.55,
    "Moral / Religious Framing":     0.50,
    "Anti-Establishment":            0.45,
    "False Authority":               0.40,
    "Unverified Claims":             0.30,
    "Neutral Reporting":             0.05,
    "ERROR":                         0.0,
}


# ─────────────────────────────────────────────
# STEP 1 — LOAD DATA
# ─────────────────────────────────────────────
def load_data(filepath: str) -> pd.DataFrame:
    print(f"[1/5] Loading master data from: {filepath}")
    df = pd.read_csv(filepath)
    df = df.dropna(subset=["title", "tactic_label"])
    df = df.reset_index(drop=True)
    print(f"      Loaded {len(df)} headlines from {df['source'].nunique()} sources.")
    return df


# ─────────────────────────────────────────────
# STEP 2 — EMBED + CLUSTER INTO EVENTS
# ─────────────────────────────────────────────
def embed_and_cluster(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (embeddings, cluster_labels).
    cluster_labels[i] == -1 means headline i is noise (unique, no cluster).
    """
    print(f"\n[2/5] Embedding {len(df)} headlines with '{EMBEDDING_MODEL}'...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = model.encode(df["title"].tolist(), show_progress_bar=True)

    print(f"      Running DBSCAN clustering (eps={DBSCAN_EPS}, min_samples={DBSCAN_MIN_SAMPLES})...")
    # Convert cosine similarity → distance matrix for DBSCAN
    cosine_dist_matrix = 1 - cosine_similarity(embeddings)
    # Clamp floating point noise
    cosine_dist_matrix = np.clip(cosine_dist_matrix, 0, 2)

    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="precomputed")
    labels = db.fit_predict(cosine_dist_matrix)

    n_events   = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise    = list(labels).count(-1)
    print(f"      Found {n_events} event clusters | {n_noise} standalone headlines.")
    return embeddings, labels


# ─────────────────────────────────────────────
# STEP 3 — BUILD EVENT OBJECTS + NARRATIVE DELTA
# ─────────────────────────────────────────────
def compute_narrative_delta(perspectives: list[str], tactics: list[str]) -> dict:
    """
    Measures how differently an event is framed across perspectives.

    Returns a dict with:
      - dominant_tactic: the most common framing label
      - agreement_score: 0.0 (total chaos) → 1.0 (all sources say the same thing)
      - framing_map: { perspective → tactic }
      - severity_score: weighted average of tactic severities in this cluster
      - delta_score: 1 - agreement_score, a direct "manipulation divergence" metric
    """
    if not tactics:
        return {}

    # Framing map: perspective → list of tactics used
    framing_map = defaultdict(list)
    for p, t in zip(perspectives, tactics):
        framing_map[p].append(t)

    # Most common tactic overall
    from collections import Counter
    tactic_counts  = Counter(tactics)
    dominant       = tactic_counts.most_common(1)[0][0]
    dominant_count = tactic_counts.most_common(1)[0][1]

    # Agreement score: fraction of headlines that use the dominant framing
    agreement_score = round(dominant_count / len(tactics), 3)

    # Severity score: average severity of all tactics in this cluster
    severity_vals  = [TACTIC_SEVERITY.get(t, 0.0) for t in tactics]
    severity_score = round(float(np.mean(severity_vals)), 3)

    # Unique tactic diversity (more = more contested narrative)
    unique_tactics = list(set(tactics))

    return {
        "dominant_tactic":  dominant,
        "agreement_score":  agreement_score,
        "delta_score":      round(1 - agreement_score, 3),
        "severity_score":   severity_score,
        "unique_tactics":   unique_tactics,
        "framing_map":      {k: list(set(v)) for k, v in framing_map.items()},
    }


def build_event_clusters(df: pd.DataFrame, labels: np.ndarray) -> list[dict]:
    """
    Groups all headlines by their cluster ID and builds rich event objects.
    Cluster -1 (noise) headlines are stored as singleton events.
    """
    print(f"\n[3/5] Building event objects and computing Narrative Deltas...")
    events = []

    unique_labels = sorted(set(labels))

    for label in unique_labels:
        mask = labels == label
        cluster_df = df[mask].copy()

        # --- THE FIX: Handle DBSCAN Noise (-1) ---
        if label == -1:
            # Treat EVERY single noise headline as its own isolated event
            for idx, row in cluster_df.iterrows():
                event = {
                    "event_id":        f"standalone_{idx}",
                    "is_event_cluster": False,
                    "headline_count":  1,
                    "representative_title": row["title"],
                    "sources":         [row["source"]],
                    "perspectives":    [row.get("perspective", "Unknown")],
                    "headlines":       [row.to_dict()],
                    "narrative_delta": compute_narrative_delta([row.get("perspective", "Unknown")], [row["tactic_label"]]),
                    "timestamp":       datetime.now().isoformat(),
                }
                events.append(event)
            continue # Skip the rest of the loop for the -1 bucket
        # -----------------------------------------

        # Normal logic for valid clusters (label 0, 1, 2...)
        headlines = cluster_df.to_dict(orient="records")
        perspectives = cluster_df["perspective"].tolist()
        tactics      = cluster_df["tactic_label"].tolist()
        sources      = cluster_df["source"].tolist()

        if "tactic_confidence" in cluster_df.columns:
            representative = cluster_df.loc[cluster_df["tactic_confidence"].idxmax(), "title"]
        else:
            representative = headlines[0]["title"]

        narrative = compute_narrative_delta(perspectives, tactics)

        event = {
            "event_id":        f"evt_{label}",
            "is_event_cluster": True,
            "headline_count":  int(mask.sum()),
            "representative_title": representative,
            "sources":         list(set(sources)),
            "perspectives":    list(set(perspectives)),
            "headlines":       headlines,
            "narrative_delta": narrative,
            "timestamp":       datetime.now().isoformat(),
        }
        events.append(event)

    # Sort: most contested (highest delta) first
    events.sort(key=lambda e: e["narrative_delta"].get("delta_score", 0), reverse=True)
    print(f"      Built {len(events)} event objects.")
    return events


# ─────────────────────────────────────────────
# STEP 4 — PERSIST TO CHROMADB
# ─────────────────────────────────────────────
def persist_to_chromadb(df: pd.DataFrame, embeddings: np.ndarray):
    """
    Upserts each headline as a vector document into ChromaDB.
    This gives Truth-Tide cross-session memory — it can recognize
    when the same event resurfaces days later under a new headline.
    """
    print(f"\n[4/5] Persisting {len(df)} vectors to ChromaDB at '{CHROMA_DB_PATH}'...")

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )

    # Get or create the collection
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}   # Use cosine distance for similarity search
    )

    # Build unique IDs: source + published_date + first 40 chars of title
    def make_id(row):
        raw = f"{row['source']}_{row.get('published_date','?')}_{row['title'][:40]}"
        return raw.replace(" ", "_").replace("/", "-")[:100]

    ids        = [make_id(row) for _, row in df.iterrows()]
    embeddings_list = embeddings.tolist()

    metadatas = [
        {
            "source":           str(row.get("source", "")),
            "perspective":      str(row.get("perspective", "")),
            "tactic_label":     str(row.get("tactic_label", "")),
            "tactic_confidence":float(row.get("tactic_confidence", 0.0)),
            "published_date":   str(row.get("published_date", "")),
            "link":             str(row.get("link", "")),
        }
        for _, row in df.iterrows()
    ]
    documents = df["title"].tolist()

    # Upsert in batches of 50 to avoid memory spikes
    BATCH = 50
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i:i+BATCH],
            embeddings=embeddings_list[i:i+BATCH],
            documents=documents[i:i+BATCH],
            metadatas=metadatas[i:i+BATCH],
        )

    total = collection.count()
    print(f"      ChromaDB now holds {total} total vectors in '{COLLECTION_NAME}'.")


def query_similar_events(query_text: str, n_results: int = 5) -> list[dict]:
    """
    Query ChromaDB to find historically similar headlines.
    Use this to check if a 'new' story is actually an old narrative resurfacing.
    """
    model = SentenceTransformer(EMBEDDING_MODEL)
    query_embedding = model.encode([query_text]).tolist()

    client = chromadb.PersistentClient(
        path=CHROMA_DB_PATH,
        settings=Settings(anonymized_telemetry=False)
    )
    collection = client.get_collection(COLLECTION_NAME)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        hits.append({
            "title":            doc,
            "similarity":       round(1 - dist, 4),   # convert distance → similarity
            "source":           meta.get("source"),
            "perspective":      meta.get("perspective"),
            "tactic_label":     meta.get("tactic_label"),
            "published_date":   meta.get("published_date"),
        })
    return hits


# ─────────────────────────────────────────────
# STEP 5 — EXPORT NARRATIVE REPORT
# ─────────────────────────────────────────────
def build_summary_stats(df: pd.DataFrame, events: list[dict]) -> dict:
    """Top-level stats for the dashboard header."""
    tactic_dist = df["tactic_label"].value_counts().to_dict()
    source_dist = df["source"].value_counts().to_dict()

    event_clusters   = [e for e in events if e["is_event_cluster"]]
    contested_events = [e for e in event_clusters if e["narrative_delta"].get("delta_score", 0) >= 0.5]

    return {
        "total_headlines":     int(len(df)),
        "total_sources":       int(df["source"].nunique()),
        "total_tactics":       int(df["tactic_label"].nunique()),
        "total_event_clusters":len(event_clusters),
        "contested_events":    len(contested_events),  # delta_score >= 0.5
        "avg_severity":        round(
            float(np.mean([TACTIC_SEVERITY.get(t, 0) for t in df["tactic_label"]])), 3
        ),
        "tactic_distribution": tactic_dist,
        "source_distribution": source_dist,
        "run_timestamp":       datetime.now().isoformat(),
    }


def export_report(events: list[dict], summary: dict, output_path: str):
    print(f"\n[5/5] Exporting Narrative Report to: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    report = {
        "meta":   summary,
        "events": events,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    size_kb = os.path.getsize(output_path) / 1024
    print(f"      Report saved. Size: {size_kb:.1f} KB")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def run_semantic_engine():
    print("=" * 56)
    print("  TRUTH-TIDE · Semantic Engine v2")
    print("=" * 56)

    df                   = load_data(INPUT_FILE)
    embeddings, labels   = embed_and_cluster(df)
    events               = build_event_clusters(df, labels)
    persist_to_chromadb(df, embeddings)
    summary              = build_summary_stats(df, events)
    export_report(events, summary, OUTPUT_REPORT)

    # ── PRINT INTEL SUMMARY ──
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
        print(f"  [{nd.get('delta_score','?'):.2f} Δ] {e['representative_title'][:65]}...")
        print(f"        Tactics seen: {', '.join(nd.get('unique_tactics', []))}")
    print("─" * 56)
    print("  Done. Run the dashboard to visualise.\n")


if __name__ == "__main__":
    run_semantic_engine()
