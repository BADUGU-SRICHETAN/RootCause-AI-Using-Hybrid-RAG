from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.root_cause_pipeline import RootCausePipeline


# --------------------------------------------------
# Project Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIRECTORY = PROJECT_ROOT / "data" / "raw"

VECTORSTORE_DIRECTORY = (
    PROJECT_ROOT / "vectorstore" / "api_investigation"
)


# --------------------------------------------------
# FastAPI Application
# --------------------------------------------------

app = FastAPI(
    title="RootCause AI",
    description="Hybrid RAG-based Software Incident Investigation System",
    version="1.0.0"
)


# --------------------------------------------------
# Pipeline Initialization
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_pipeline() -> RootCausePipeline:

    return RootCausePipeline(
        data_directory=str(DATA_DIRECTORY),
        persist_directory=str(VECTORSTORE_DIRECTORY),
        model_name="gpt-4o-mini"
    )


# --------------------------------------------------
# Request Schema
# --------------------------------------------------

class InvestigationRequest(BaseModel):

    query: str = Field(
        ...,
        min_length=10,
        max_length=2000,
        description="Description of the software incident"
    )

    top_k: int = Field(
        default=3,
        ge=1,
        le=10
    )

    candidate_k: int = Field(
        default=5,
        ge=1,
        le=20
    )


# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "application": "RootCause AI",
        "pipeline_initialized": get_pipeline.cache_info().currsize > 0
    }


# --------------------------------------------------
# Incident Investigation
# --------------------------------------------------

@app.post("/investigate")
def investigate_incident(
    request: InvestigationRequest
) -> dict[str, Any]:

    if request.candidate_k < request.top_k:
        raise HTTPException(
            status_code=422,
            detail="candidate_k must be greater than or equal to top_k."
        )

    try:

        pipeline = get_pipeline()

        result = pipeline.run(
            query=request.query,
            top_k=request.top_k,
            candidate_k=request.candidate_k
        )

        return {
            "status": "success",
            "application": "RootCause AI",
            "investigation": result
        }

    except HTTPException:
        raise

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=f"Investigation failed: {str(error)}"
        ) from error