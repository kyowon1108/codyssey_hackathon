"""
Viewer page API endpoints
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from pathlib import Path

from app.config import settings
from app.db import crud
from app.db.database import SessionLocal
from app.utils.logger import setup_logger
from app.utils.colmap_reader import get_camera_position_for_viewer

logger = setup_logger(__name__)
router = APIRouter(tags=["viewer"])


@router.get("/v/{pub_key}")
async def view_result(pub_key: str):
    """
    Redirect to PlayCanvas Model Viewer with PLY file and camera position

    Args:
        pub_key: Public key

    Returns:
        Redirect to viewer with PLY URL and camera position from first image (rotated 180°)
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

        # Get camera position from first COLMAP image (rotated 180 degrees)
        job_work_dir = Path(settings.DATA_DIR) / job.job_id / "work"
        camera_pos = get_camera_position_for_viewer(job_work_dir, rotate_180=True)

        # Build viewer URL with camera position
        if camera_pos:
            x, y, z = camera_pos
            viewer_url = f"/viewer/?load={ply_url}&cameraPosition={x:.3f},{y:.3f},{z:.3f}"
            logger.info(f"Viewer URL for {pub_key}: {viewer_url} (camera from first image, rotated 180°)")
        else:
            # Fallback to default view if camera position not available
            viewer_url = f"/viewer/?load={ply_url}"
            logger.warning(f"Could not read camera position for {pub_key}, using default view")

        return RedirectResponse(url=viewer_url)
    finally:
        db.close()
