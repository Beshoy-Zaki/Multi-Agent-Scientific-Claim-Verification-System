"""FastAPI application entrypoint exposing MASCV endpoints."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from ui.backend.routes import upload, claims, debate, reports
from ui.backend.session_manager import list_sessions

app = FastAPI(
    title="MASCV API",
    description="Backend API for Multi-Agent Scientific Claim Verification System",
    version="0.1.0",
)

# Enable CORS for local dashboards and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(upload.router, prefix="/api/upload", tags=["Upload"])
app.include_router(claims.router, prefix="/api/claims", tags=["Claims"])
app.include_router(debate.router, prefix="/api/debate", tags=["Debate"])
app.include_router(reports.router, prefix="/api/reports", tags=["Reports"])


@app.get("/")
def health_check():
    """Health check endpoint for API status monitoring."""
    return {"status": "ok", "service": "MASCV API"}


@app.get("/api/sessions")
def get_active_sessions():
    """List all currently loaded investigation sessions."""
    return {"sessions": list_sessions()}

