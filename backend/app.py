import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from analysis_orchestrator import UserFacingAnalysisError, run_analysis, stream_analysis
from api_contract import AnalyzeResponse, ClaimRequest

app = FastAPI(title="Dual-Dimension Misinformation Analyzer API")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_ROOT = os.path.join(PROJECT_ROOT, "frontend")
ASSETS_ROOT = os.path.join(FRONTEND_ROOT, "assets")

if os.path.exists(ASSETS_ROOT):
    app.mount("/assets", StaticFiles(directory=ASSETS_ROOT), name="assets")


@app.get("/", include_in_schema=False)
def serve_frontend() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_ROOT, "index.html"), headers={"Cache-Control": "no-store"})


@app.get("/style.css", include_in_schema=False)
def serve_frontend_styles() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_ROOT, "style.css"), media_type="text/css", headers={"Cache-Control": "no-store"})


@app.get("/script.js", include_in_schema=False)
def serve_frontend_script() -> FileResponse:
    return FileResponse(os.path.join(FRONTEND_ROOT, "script.js"), media_type="application/javascript", headers={"Cache-Control": "no-store"})


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_claim(request: ClaimRequest) -> AnalyzeResponse:
    try:
        return run_analysis(request)
    except UserFacingAnalysisError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/analyze/stream")
def analyze_claim_stream(request: ClaimRequest) -> StreamingResponse:
    return StreamingResponse(
        stream_analysis(request),
        media_type="text/event-stream",
    )
