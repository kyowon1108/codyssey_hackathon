"""
Viewer page API endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.database import SessionLocal
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(tags=["viewer"])


@router.get("/v/{pub_key}")
async def view_result(pub_key: str):
    """
    Redirect to PlayCanvas Model Viewer with PLY file

    Args:
        pub_key: Public key

    Returns:
        Redirect to viewer with PLY URL
    """
    db = SessionLocal()
    try:
        job = crud.get_job_by_pub_key(db, pub_key)
        if not job:
            raise HTTPException(404, "Job not found")

        if job.status != "COMPLETED":
            raise HTTPException(400, f"Job not completed yet. Current status: {job.status}")

        # Build PLY file URL for PlayCanvas viewer
        ply_url = f"{settings.BASE_URL}/recon/pub/{pub_key}/cloud.ply"

        # Redirect to PlayCanvas viewer with load parameter
        viewer_url = f"/viewer/?load={ply_url}"

        return RedirectResponse(url=viewer_url)
    finally:
        db.close()
