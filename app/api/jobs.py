"""
Job management API endpoints
"""
import asyncio
import secrets
import string
from pathlib import Path
from typing import List
from fastapi import APIRouter, Depends, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import crud
from app.db.database import SessionLocal
from app.schemas.job import JobCreateResponse, JobStatusResponse, JobListResponse
from app.utils.image import validate_image_file, save_image
from app.utils.logger import setup_logger
from app.core.colmap import COLMAPPipeline
from app.core.gaussian_splatting import GaussianSplattingTrainer

logger = setup_logger(__name__)
router = APIRouter(prefix="/recon", tags=["reconstruction"])

# Job processing semaphore
job_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def generate_job_id() -> str:
    """Generate 8-character alphanumeric job ID"""
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(8))


def generate_pub_key() -> str:
    """Generate 10-character alphanumeric public key"""
    chars = string.ascii_letters + string.digits
    return ''.join(secrets.choice(chars) for _ in range(10))


@router.post("/jobs", response_model=JobCreateResponse)
async def create_job(
    files: List[UploadFile] = File(...),
    original_resolution: bool = Form(False)
):
    """
    Create new reconstruction job

    Args:
        files: List of uploaded image files
        original_resolution: Whether to use original image resolution

    Returns:
        Job creation response with job_id and pub_key
    """
    # Validate image count
    if len(files) < settings.MIN_IMAGES:
        raise HTTPException(400, f"At least {settings.MIN_IMAGES} images required")
    if len(files) > settings.MAX_IMAGES:
        raise HTTPException(400, f"Maximum {settings.MAX_IMAGES} images allowed")

    # Validate all images
    for file in files:
        validate_image_file(file)

    # Generate unique IDs
    job_id = generate_job_id()
    pub_key = generate_pub_key()

    # Create job directory
    job_dir = settings.DATA_DIR / job_id
    upload_dir = job_dir / "upload" / "images"
    upload_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Created job {job_id} with {len(files)} images")

    # Save images
    for file in files:
        output_path = upload_dir / file.filename
        await save_image(file, output_path, resize=not original_resolution)

    # Create database record
    db = SessionLocal()
    try:
        crud.create_job(
            db=db,
            job_id=job_id,
            pub_key=pub_key,
            image_count=len(files),
            original_resolution=original_resolution
        )
        db.commit()
    finally:
        db.close()

    # Start background processing
    asyncio.create_task(process_job(job_id, original_resolution))

    return JobCreateResponse(
        job_id=job_id,
        pub_key=pub_key,
        original_resolution=original_resolution
    )


@router.get("/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Get job status and logs

    Args:
        job_id: Job identifier

    Returns:
        Job status response
    """
    db = SessionLocal()
    try:
        job = crud.get_job_by_id(db, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        # Read last 50 lines of log
        log_file = settings.DATA_DIR / job_id / "logs" / "process.log"
        log_tail = []
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()
                log_tail = [line.rstrip() for line in lines[-50:]]

        # Build viewer URL if completed
        viewer_url = None
        if job.status == "COMPLETED":
            viewer_url = f"{settings.BASE_URL}/v/{job.pub_key}"

        return JobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            log_tail=log_tail,
            gaussian_count=job.gaussian_count,
            viewer_url=viewer_url,
            error_message=job.error_message,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at
        )
    finally:
        db.close()


@router.get("/pub/{pub_key}/cloud.ply")
async def get_point_cloud(pub_key: str):
    """
    Download point cloud PLY file

    Args:
        pub_key: Public key

    Returns:
        PLY file
    """
    db = SessionLocal()
    try:
        job = crud.get_job_by_pub_key(db, pub_key)
        if not job:
            raise HTTPException(404, "Job not found")

        if job.status != "COMPLETED":
            raise HTTPException(400, "Job not completed yet")

        # Try new structure first
        ply_file = settings.DATA_DIR / job.job_id / "output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}" / "point_cloud_filtered.ply"

        if not ply_file.exists():
            # Fallback to unfiltered (new structure)
            ply_file = settings.DATA_DIR / job.job_id / "output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}" / "point_cloud.ply"

        # Try old structure (gs_output)
        if not ply_file.exists():
            ply_file = settings.DATA_DIR / job.job_id / "gs_output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}" / "point_cloud_filtered.ply"

        if not ply_file.exists():
            ply_file = settings.DATA_DIR / job.job_id / "gs_output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}" / "point_cloud.ply"

        if not ply_file.exists():
            raise HTTPException(404, "Point cloud file not found")

        return FileResponse(
            path=str(ply_file),
            media_type="application/octet-stream",
            filename="point_cloud.ply"
        )
    finally:
        db.close()


@router.get("/pub/{pub_key}/scene.splat")
async def get_splat_file(pub_key: str):
    """
    Download splat file for viewer

    Args:
        pub_key: Public key

    Returns:
        Splat file
    """
    db = SessionLocal()
    try:
        job = crud.get_job_by_pub_key(db, pub_key)
        if not job:
            raise HTTPException(404, "Job not found")

        if job.status != "COMPLETED":
            raise HTTPException(400, "Job not completed yet")

        # Try new structure first
        splat_file = settings.DATA_DIR / job.job_id / "output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}" / "scene.splat"

        # Try old structure (gs_output)
        if not splat_file.exists():
            splat_file = settings.DATA_DIR / job.job_id / "gs_output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}" / "scene.splat"

        if not splat_file.exists():
            raise HTTPException(404, "Splat file not found")

        return FileResponse(
            path=str(splat_file),
            media_type="application/octet-stream",
            filename="scene.splat"
        )
    finally:
        db.close()


async def process_job(job_id: str, original_resolution: bool):
    """
    Background job processing pipeline

    Args:
        job_id: Job identifier
        original_resolution: Whether to use original resolution
    """
    async with job_semaphore:
        db = SessionLocal()
        job_dir = settings.DATA_DIR / job_id
        log_dir = job_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file_path = log_dir / "process.log"

        try:
            # Update status to PROCESSING
            crud.update_job_status(db, job_id, "PROCESSING")
            db.commit()

            with open(log_file_path, 'w') as log_file:
                log_file.write(f">> [Job {job_id}] Starting reconstruction pipeline\n")
                log_file.flush()

                # Initialize COLMAP pipeline
                colmap = COLMAPPipeline(job_dir)

                # Step 1: Feature extraction
                log_file.write(">> [COLMAP] Extracting features...\n")
                log_file.flush()
                colmap.database_path.parent.mkdir(parents=True, exist_ok=True)
                await colmap.extract_features(log_file)

                # Step 2: Feature matching
                log_file.write(">> [COLMAP] Matching features...\n")
                log_file.flush()
                await colmap.match_features(log_file)

                # Step 3: Sparse reconstruction
                log_file.write(">> [COLMAP] Reconstructing sparse model...\n")
                log_file.flush()
                model_path = await colmap.reconstruct(log_file)

                # Step 4: Undistort images
                log_file.write(">> [COLMAP] Undistorting images...\n")
                log_file.flush()
                work_dir = await colmap.undistort_images(model_path, log_file)

                # Step 5: Convert to text format
                log_file.write(">> [COLMAP] Converting to text format...\n")
                log_file.flush()
                await colmap.convert_to_text(work_dir / "sparse" / "0", log_file)

                # Step 6: Gaussian Splatting training
                log_file.write(">> [GS] Starting Gaussian Splatting training...\n")
                log_file.flush()

                output_dir = job_dir / "output"
                gs_trainer = GaussianSplattingTrainer(work_dir, output_dir)
                iteration_dir = await gs_trainer.train(log_file)

                # Step 7: Post-processing
                log_file.write(">> [GS] Post-processing results...\n")
                log_file.flush()
                gs_trainer.post_process(iteration_dir, log_file)

                # Count Gaussians
                ply_file = iteration_dir / "point_cloud_filtered.ply"
                if not ply_file.exists():
                    ply_file = iteration_dir / "point_cloud.ply"

                gaussian_count = 0
                if ply_file.exists():
                    with open(ply_file, 'rb') as f:
                        for line in f:
                            line_str = line.decode('utf-8', errors='ignore')
                            if line_str.startswith("element vertex"):
                                gaussian_count = int(line_str.split()[-1])
                                break

                # Update job as completed
                crud.update_job_status(
                    db, job_id, "COMPLETED",
                    gaussian_count=gaussian_count
                )
                db.commit()

                log_file.write(f">> [SUCCESS] Job completed! Generated {gaussian_count} Gaussians\n")
                log_file.flush()
                logger.info(f"Job {job_id} completed successfully")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")

            # Log error to database
            crud.log_error(
                db, job_id,
                stage="PIPELINE",
                error_type=type(e).__name__,
                error_message=str(e)
            )
            crud.update_job_status(
                db, job_id, "FAILED",
                error_message=str(e)
            )
            db.commit()

            # Write to log file
            if log_file_path.exists():
                with open(log_file_path, 'a') as log_file:
                    log_file.write(f"\n>> [ERROR] {str(e)}\n")
        finally:
            db.close()
