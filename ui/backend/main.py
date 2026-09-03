"""FastAPI application entrypoint exposing MASCV endpoints."""

from fastapi import FastAPI
from ui.backend.routes import upload, claims, debate, reports

app = FastAPI(
    title="MASCV API",
    description="Backend API for Multi-Agent Scientific Claim Verification System",
    version="0.1.0",
)

app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(claims.router, prefix="/api/claims", tags=["Claims"])
app.include_router(debate.router, prefix="/api/debate", tags=["Debate"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


@app.get("/")
def health_check():
    return {"status": "ok", "service": "MASCV API"}
