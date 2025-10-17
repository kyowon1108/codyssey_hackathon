"""
Viewer page API endpoints
"""
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.database import SessionLocal
from app.utils.logger import setup_logger

logger = setup_logger(__name__)
router = APIRouter(tags=["viewer"])

templates = Jinja2Templates(directory=str(settings.BASE_DIR / "templates"))


@router.get("/v/{pub_key}", response_class=HTMLResponse)
async def view_result(request: Request, pub_key: str):
    """
    Render viewer page for completed job

    Args:
        request: FastAPI request object
        pub_key: Public key

    Returns:
        HTML viewer page
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

        return templates.TemplateResponse(
            "viewer.html",
            {
                "request": request,
                "pub_key": pub_key,
                "ply_url": ply_url,
                "gaussian_count": job.gaussian_count or 0,
                "psnr": job.psnr,
                "ssim": job.ssim,
                "lpips": job.lpips
            }
        )
    finally:
        db.close()
