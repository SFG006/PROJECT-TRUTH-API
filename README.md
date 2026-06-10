# PROJECT TRUTH: Autonomous OSINT & Narrative Engine

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg?style=for-the-badge&logo=python)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](#)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.1-F9AB00?style=for-the-badge)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-darkred.svg?style=for-the-badge)](#)

**Project Truth** is an automated Open Source Intelligence (OSINT) API and data pipeline designed for live geopolitical narrative and propaganda tactics tracking. 

By continuously scraping institutional global news feeds, the engine utilizes a high-speed Groq LLM pipeline to classify psychological manipulation tactics, followed by semantic clustering to map conflicting global narratives in 3D space.

<img width="1948" height="954" alt="FireShot Capture 003 - Project Truth Command Center - sfg006-project-truth hf space" src="https://github.com/user-attachments/assets/b9e00900-eaf3-40ef-a358-38d5db47b1e0" />


---

##  Live Intelligence Dashboard  

The platform is automatically deployed and updated daily. Access both the API and UI below:

###  Backend API  
[![Backend API](https://img.shields.io/badge/Project_Truth_API-Open-1f6feb?style=for-the-badge&logo=fastapi)](https://sfg006-project-truth-api.hf.space)

###  Frontend Dashboard  
[![Frontend Dashboard](https://img.shields.io/badge/Project_Truth_Dashboard-Launch-005571?style=for-the-badge&logo=vercel)](https://sfg006-project-truth.hf.space)

## Pipeline Overview

The platform operates autonomously through a strictly institutional, three-stage data pipeline:

### 1. Data Siphoning (Ingestion & Translation)
The engine pulls high-volume text from diverse geopolitical perspectives, automatically translating non-English sources into English using `deep-translator`.
* **Multilingual RSS Scraper (`data_siphon.py`):** Parses full article bodies from global sources, including Western wire services (Reuters, BBC), Middle Eastern outlets (Al Jazeera, Tehran Times), Russian state media (RT, TASS), and North American feeds (CNN, Fox).
* **Paywall/Bot-Blocker Resilience:** Automatically falls back to translated RSS summaries if the main article body is unreachable.

### 2. Tactical Analysis Engine (`tactic_engine.py`)
Utilizes **Groq's `llama-3.1-8b-instant`** for high-speed, zero-shot inference.
* Analyzes institutional article framing to assign broad propaganda labels (e.g., *Fear-Mongering, Disinformation, Victim-Blaming*).
* Inventively generates specific 2-to-4 word micro-tactics for granular tracking (e.g., "Nuclear Threat Exaggeration", "Historical Revisionism").

### 3. Semantic Engine (`semantic_engine.py`)
Processes the classified data to map the "shape" of the global information war.
* **Embeddings:** Generates sentence embeddings using `SentenceTransformer('all-MiniLM-L6-v2')`.
* **Event Clustering:** Applies **DBSCAN** to automatically group overlapping headlines into unique "events" based on cosine distance.
* **3D Spatial Mapping:** Utilizes **UMAP** for dimensionality reduction, generating specific X, Y, Z coordinates for frontend 3D UI rendering.
* **Narrative Delta:** Calculates an agreement/severity score to highlight heavily contested geopolitical events.

### 4. RESTful API
A **FastAPI** backend that serves the processed intelligence to the downstream dashboard UI.

---

##   API Routes

The FastAPI server exposes the following data streams:

| Endpoint | Method | Description |
| :--- | :---: | :--- |
| `/api/narratives` | `GET` | Returns clustered event objects, 3D UMAP coordinates, and Narrative Delta scores. |
| `/api/headlines` | `GET` | Returns the most recently siphoned headlines and their specific Groq-classified tactics. |
| `/api/tactics-summary` | `GET` | Aggregated distribution of psychological tactics detected in the current cycle. |
| `/api/sources` | `GET` | Breakdown of which news networks/sources are deploying specific manipulation tactics. |

---

## Getting Started

### Prerequisites
* Python 3.11+
* Groq API Key

### 1. Clone the Repository
```bash
git clone [https://github.com/SFG006/PROJECT-TRUTH-API.git](https://github.com/SFG006/PROJECT-TRUTH-API.git)
cd Truth-Tide
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Environment Configuration
Create a `.env` file in the root directory and add your Groq credentials:
```env
# Groq Inference
GROQ_API_KEY=your_groq_api_key_here
```

### 4. Run the Pipeline & API
Run the data collector to pull fresh news:
```bash
python data_siphon.py
```
Run the tactical classification and semantic clustering:
```bash
python tactic_engine.py
```
Start the FastAPI server:
```bash
uvicorn main:app --reload --port 7860
```

---

##  Automated Deployment (CI/CD)

Truth-Tide utilizes a fully automated GitHub Actions pipeline (`pipeline.yml`). 

Every day at Midnight (UTC), or upon a manual push to the `main` branch, the system will:
1. Spin up an Ubuntu runner.
2. Execute the data siphon and tactic engines.
3. Commit the newly generated JSON intelligence reports.
4. Push the synchronized repository directly to Hugging Face Spaces for live UI rendering.

## Docker Deployment

Truth-Tide is containerized for easy deployment to cloud platforms. The included `Dockerfile` uses a secure non-root user and explicitly configures Uvicorn to trust reverse proxies.
```bash
docker build -t project-truth-api .
docker run -p 7860:7860 --env-file .env project-truth-api
```

---
<div align="center">
  <sub><i>"The first casualty of war is the truth. The second is the infrastructure that maps it."</i></sub>
</div>
