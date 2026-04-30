from fastapi import FastAPI, HTTPException
import pandas as pd
import glob
import os
import json

app = FastAPI(
    title="Project Truth OSINT API",
    description="Live geopolitical narrative and propaganda tactics tracking.",
    version="2.0.0"
)


def load_latest_data():
    """Directly loads the single master intelligence file."""
    file_path = "data/processed/master_tactics_latest.csv"

    if not os.path.exists(file_path):
        return None

    # Read the file and fill empty cells so JSON doesn't crash
    df = pd.read_csv(file_path).fillna("N/A")
    return df


@app.get("/")
def read_root():
    return {
        "status": "online",
        "message": "Welcome to the Project Truth API. Go to /docs for the interactive dashboard."
    }


@app.get("/api/narratives")
def get_narratives():
    """Serves the clustered event data and Narrative Delta scores."""
    file_path = "data/processed/narrative_report.json"

    if not os.path.exists(file_path):
        return {"meta": {}, "events": []}

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        return {"error": f"Failed to load narratives: {str(e)}"}


@app.get("/api/headlines")
def get_all_headlines(limit: int = 50):
    """Returns the most recent siphoned headlines and their tactical analysis."""
    df = load_latest_data()
    if df is None:
        raise HTTPException(status_code=404, detail="Intelligence data not found.")

    # Convert dataframe to a list of dictionaries for JSON output
    data = df.head(limit).to_dict(orient="records")
    return {"count": len(data), "data": data}


@app.get("/api/tactics-summary")
def get_tactics_summary():
    """Returns an aggregated count of which tactics are being used today."""
    df = load_latest_data()
    if df is None:
        raise HTTPException(status_code=404, detail="Intelligence data not found.")

    # Count how many times each tactic appears
    summary = df['tactic_label'].value_counts().to_dict()
    return {"tactics_distribution": summary}


@app.get("/api/sources")
def get_source_breakdown():
    """Returns a breakdown of tactics used by specific news/social networks."""
    df = load_latest_data()
    if df is None:
        raise HTTPException(status_code=404, detail="Intelligence data not found.")

    # Group by source and tactic
    grouped = df.groupby(['source', 'tactic_label']).size().unstack(fill_value=0).to_dict(orient="index")
    return {"source_tactics": grouped}