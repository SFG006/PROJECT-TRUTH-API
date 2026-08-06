from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field, RootModel
import pandas as pd
import os
import aiofiles
import json
from typing import List, Dict, Union
from fastapi.concurrency import run_in_threadpool


# ─────────────────────────────────────────────
# PYDANTIC V2 SCHEMAS (Powers the Swagger /docs)
# ─────────────────────────────────────────────

class TacticDistribution(RootModel[Dict[str, int]]):
    model_config = {
        "json_schema_extra": {
            "example": {
                "Neutral Reporting": 210,
                "Fear Mongering / Alarmism": 105,
                "Disinformation": 65
            }
        }
    }

class SourceDistribution(RootModel[Dict[str, Dict[str, int]]]):
    model_config = {
        "json_schema_extra": {
            "example": {
                "CNN": 150,
                "RT": 120,
                "Al Jazeera": 90
            }
        }
    }


class MetaStats(BaseModel):
    total_headlines: int = Field(..., description="Total headlines analyzed across all pipelines.", examples=[500])
    total_sources: int = Field(..., description="Number of unique media networks monitored.", examples=[12])
    total_tactics: int = Field(..., description="Distinct types of analytical manipulation vectors identified.",
                               examples=[6])
    total_event_clusters: int = Field(..., description="Cohesive narrative clusters discovered by DBSCAN.",
                                      examples=[35])
    contested_events: int = Field(..., description="Events where Narrative Delta exceeds the 0.50 threshold.",
                                  examples=[4])
    avg_severity: float = Field(..., description="Global media hostility index average (0.0 to 1.0).", examples=[0.425])
    tactic_distribution: Dict[str, int] = Field(..., description="Global count of distinct tactical behaviors.")
    source_distribution: Dict[str, int] = Field(..., description="Global breakdown of data contributions per source.")
    run_timestamp: str = Field(..., description="ISO timestamp of the backend orchestration run.",
                               examples=["2026-06-19T12:00:00"])


class NarrativeDeltaSchema(BaseModel):
    dominant_tactic: str = Field(..., description="The most frequent tactic found in this cluster.",
                                 examples=["Fear Mongering"])
    agreement_score: float = Field(..., description="Ratio of consensus around the dominant tactic.", examples=[0.25])
    delta_score: float = Field(..., description="Narrative divergence index. Higher means heavier split/spin.",
                               examples=[0.75])
    severity_score: float = Field(..., description="Average danger or aggression score of the cluster tactics.",
                                  examples=[0.725])
    unique_tactics: List[str] = Field(..., description="All distinct tactics identified within the event group.")
    framing_map: Dict[str, List[str]] = Field(...,
                                              description="Mapping showing exactly which tactics each side deployed.")


class HeadlineSchema(BaseModel):
    title: str = Field(..., description="The extracted headline string.",
                       examples=["Cyberattack forces regional bank offline"])
    source: str = Field(..., description="The publishing node or media entity.", examples=["RT"])
    perspective: str = Field(..., description="The calculated political or regional alignment.", examples=["Eastern"])
    tactic_label: str = Field(..., description="The classified pipeline tactic label.", examples=["Disinformation"])
    x: float = Field(..., description="UMAP 3D Space coordinate X.", examples=[1.12])
    y: float = Field(..., description="UMAP 3D Space coordinate Y.", examples=[-0.54])
    z: float = Field(..., description="UMAP 3D Space coordinate Z.", examples=[3.41])


class EventClusterSchema(BaseModel):
    event_id: str = Field(..., description="Unique event identification token.", examples=["evt_0"])
    is_event_cluster: bool = Field(..., description="Flags if this is an organized cluster or standalone noise.",
                                   examples=[True])
    headline_count: int = Field(..., description="Number of active articles grouped into this specific narrative.",
                                examples=[4])
    representative_title: str = Field(..., description="The representative title selected for dashboard display.",
                                      examples=["Cyberattack hits bank"])
    sources: List[str] = Field(..., description="Deduplicated list of sources talking about this event.")
    perspectives: List[str] = Field(..., description="Deduplicated list of structural frameworks involved.")
    headlines: List[HeadlineSchema] = Field(...,
                                            description="The collection of full baseline records inside this event cluster.")
    narrative_delta: NarrativeDeltaSchema = Field(...,
                                                  description="The granular delta and tension breakdown of the event cluster.")
    timestamp: str = Field(..., description="Generation time context.", examples=["2026-06-19T12:02:00"])


class NarrativeReportResponse(BaseModel):
    meta: MetaStats = Field(..., description="Global execution and dataset stats overview.")
    events: List[EventClusterSchema] = Field(...,
                                             description="Chronological and delta sorted narrative cluster objects.")


# ─────────────────────────────────────────────
# APPLICATION INITIALIZATION
# ─────────────────────────────────────────────

app = FastAPI(
    title="Project Truth Intelligence Core API",
    description=(
        "## Real Time Geopolitical Narrative Tracking & Propaganda Metrics\n\n"
        "This API unifies downstream processing tasks including raw headline ingest, embedding vectors, "
        "DBSCAN clustering, and 3D coordinate synthesis. It provides systematic access to operational analytics "
        "tracking computational manipulation across global news vectors."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)


async def load_latest_data() -> Union[pd.DataFrame, None]:
    file_path = "data/processed/master_tactics_latest.csv"
    if not os.path.exists(file_path):
        return None
    df = await run_in_threadpool(pd.read_csv, file_path)
    return df.fillna("N/A")


# ─────────────────────────────────────────────
# API ENDPOINTS
# ─────────────────────────────────────────────

@app.get("/", tags=["System Status"])
def read_root():
    """Validates api availability and basic connection state checks."""
    return {
        "status": "online",
        "message": "Welcome to the PROJECT TRUTH Intelligence API Core. Navigate to /docs for the comprehensive dashboard interface."
    }


@app.get(
    "/api/narratives",
    response_model=NarrativeReportResponse,
    tags=["Core Intelligence Engine"],
    summary="Fetch complete clustered narratives and delta scores",
    response_description="A structured JSON object containing daily meta analysis and detailed event profiles."
)
async def get_narratives():
    """
    Renders fully integrated clusters computed by our unsupervised learning steps.
    Returns a sorted array where the stories showing the **highest Narrative Delta Scores** (maximum spin and contrasting perspectives) are surfaced at the absolute top of the stack.
    """
    file_path = "data/processed/narrative_report.json"

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Processed narrative intelligence data package not found.")

    try:
        async with aiofiles.open(file_path, "r", encoding="utf-8") as f:
            content = await f.read()
            return json.loads(content)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to decode system file index mapping: {str(e)}")


@app.get(
    "/api/headlines",
    tags=["Raw Intelligence Feeds"],
    summary="Get siphoned tracking records and custom coordinates",
    response_description="Returns count tracking along with direct array records."
)
async def get_all_headlines(
        limit: int = Query(default=100, description="Slice window limit parameter to constrain response sizes.", ge=1,
                           le=1000)
):
    """
    Returns single flattened records directly from our unified persistent staging layout.
    Useful for inspecting individual classifications or pulling raw geometry points (`x, y, z`)
    for direct custom front-end bindings.
    """
    df = await load_latest_data()
    if df is None:
        raise HTTPException(status_code=404, detail="Intelligence framework database records not found.")

    data = df.head(limit).to_dict(orient="records")
    return {"count": len(data), "data": data}


@app.get(
    "/api/tactics_summary",
    response_model=TacticDistribution,
    tags=["Global System Analytics"],
    summary="Aggregated volume tracking for narrative manipulation categories"
)
async def get_tactics_summary():
    """
    Scans the live operational data structures to output clean, total calculation summaries
    of every observed trick label distribution today. Perfect for feeding standalone bar charts.
    """
    df = await load_latest_data()
    if df is None:
        raise HTTPException(status_code=404, detail="Intelligence framework database records not found.")

    summary = df['tactic_label'].value_counts().to_dict()
    return summary


@app.get(
    "/api/sources",
    response_model=SourceDistribution,
    tags=["Global System Analytics"],
    summary="Cross tabulated framework matrix tracking behavior metrics per outlet"
)
async def get_source_breakdown():
    """
    Builds an analytical pivot tracking matrix evaluating structural behavioral profiles
    across all monitored publishers. Maps out exactly how frequently specific nodes lean into
    defined tracking tags.
    """
    df = await load_latest_data()
    if df is None:
        raise HTTPException(status_code=404, detail="Intelligence framework database records not found.")

    grouped = df.groupby(['source', 'tactic_label']).size().unstack(fill_value=0).to_dict(orient="index")
    return grouped