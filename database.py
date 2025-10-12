"""
Database models and connection management for Gaussian Splatting API
"""
from sqlalchemy import create_engine, Column, String, DateTime, Integer, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
from pathlib import Path
import json
from typing import Optional, Dict, Any

# Database setup
DATABASE_URL = "sqlite:///./gaussian_splatting.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Job(Base):
    """Job model for tracking reconstruction tasks"""
    __tablename__ = "jobs"

    job_id = Column(String(8), primary_key=True, index=True)
    pub_key = Column(String(10), unique=True, index=True, nullable=False)
    status = Column(String(20), nullable=False, default="PENDING")  # PENDING, RUNNING, DONE, FAILED

    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)

    # Configuration
    original_resolution = Column(Boolean, default=False)
    image_count = Column(Integer, default=0)
    iterations = Column(Integer, default=10000)

    # Results
    gaussian_count = Column(Integer, nullable=True)
    filtered_count = Column(Integer, nullable=True)
    removed_count = Column(Integer, nullable=True)
    file_size_mb = Column(Float, nullable=True)

    # Error tracking
    error_message = Column(Text, nullable=True)
    error_stage = Column(String(50), nullable=True)  # upload, colmap, training, filtering, conversion
    retry_count = Column(Integer, default=0)

    # Metadata
    colmap_registered_images = Column(Integer, nullable=True)
    colmap_points = Column(Integer, nullable=True)
    processing_time_seconds = Column(Float, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "job_id": self.job_id,
            "pub_key": self.pub_key,
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "original_resolution": self.original_resolution,
            "image_count": self.image_count,
            "iterations": self.iterations,
            "gaussian_count": self.gaussian_count,
            "filtered_count": self.filtered_count,
            "removed_count": self.removed_count,
            "file_size_mb": self.file_size_mb,
            "error_message": self.error_message,
            "error_stage": self.error_stage,
            "retry_count": self.retry_count,
            "colmap_registered_images": self.colmap_registered_images,
            "colmap_points": self.colmap_points,
            "processing_time_seconds": self.processing_time_seconds,
        }


class ErrorLog(Base):
    """Error log for tracking failures"""
    __tablename__ = "error_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String(8), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    stage = Column(String(50), nullable=False)
    error_type = Column(String(100), nullable=False)
    error_message = Column(Text, nullable=False)
    traceback = Column(Text, nullable=True)

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary"""
        return {
            "id": self.id,
            "job_id": self.job_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "stage": self.stage,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "traceback": self.traceback,
        }


def init_db():
    """Initialize database tables"""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# CRUD operations
def create_job(
    db: Session,
    job_id: str,
    pub_key: str,
    image_count: int = 0,
    original_resolution: bool = False,
    iterations: int = 10000
) -> Job:
    """Create a new job in database"""
    job = Job(
        job_id=job_id,
        pub_key=pub_key,
        status="PENDING",
        image_count=image_count,
        original_resolution=original_resolution,
        iterations=iterations,
        created_at=datetime.utcnow()
    )
    db.add(job)
    db.commit()
    db.refresh(job)
    return job


def get_job_by_id(db: Session, job_id: str) -> Optional[Job]:
    """Get job by job_id"""
    return db.query(Job).filter(Job.job_id == job_id).first()


def get_job_by_pub_key(db: Session, pub_key: str) -> Optional[Job]:
    """Get job by public key"""
    return db.query(Job).filter(Job.pub_key == pub_key).first()


def update_job_status(
    db: Session,
    job_id: str,
    status: str,
    error_message: Optional[str] = None,
    error_stage: Optional[str] = None
) -> Optional[Job]:
    """Update job status"""
    job = get_job_by_id(db, job_id)
    if not job:
        return None

    job.status = status

    if status == "RUNNING" and not job.started_at:
        job.started_at = datetime.utcnow()
    elif status in ["DONE", "FAILED"]:
        job.completed_at = datetime.utcnow()
        if job.started_at:
            job.processing_time_seconds = (job.completed_at - job.started_at).total_seconds()

    if error_message:
        job.error_message = error_message
    if error_stage:
        job.error_stage = error_stage

    db.commit()
    db.refresh(job)
    return job


def update_job_results(
    db: Session,
    job_id: str,
    gaussian_count: Optional[int] = None,
    filtered_count: Optional[int] = None,
    removed_count: Optional[int] = None,
    file_size_mb: Optional[float] = None,
    colmap_registered_images: Optional[int] = None,
    colmap_points: Optional[int] = None
) -> Optional[Job]:
    """Update job results"""
    job = get_job_by_id(db, job_id)
    if not job:
        return None

    if gaussian_count is not None:
        job.gaussian_count = gaussian_count
    if filtered_count is not None:
        job.filtered_count = filtered_count
    if removed_count is not None:
        job.removed_count = removed_count
    if file_size_mb is not None:
        job.file_size_mb = file_size_mb
    if colmap_registered_images is not None:
        job.colmap_registered_images = colmap_registered_images
    if colmap_points is not None:
        job.colmap_points = colmap_points

    db.commit()
    db.refresh(job)
    return job


def increment_retry_count(db: Session, job_id: str) -> Optional[Job]:
    """Increment retry count for a job"""
    job = get_job_by_id(db, job_id)
    if not job:
        return None

    job.retry_count += 1
    db.commit()
    db.refresh(job)
    return job


def log_error(
    db: Session,
    job_id: str,
    stage: str,
    error_type: str,
    error_message: str,
    traceback: Optional[str] = None
) -> ErrorLog:
    """Log an error"""
    error_log = ErrorLog(
        job_id=job_id,
        stage=stage,
        error_type=error_type,
        error_message=error_message,
        traceback=traceback,
        timestamp=datetime.utcnow()
    )
    db.add(error_log)
    db.commit()
    db.refresh(error_log)
    return error_log


def get_recent_errors(db: Session, job_id: str, limit: int = 10):
    """Get recent errors for a job"""
    return db.query(ErrorLog).filter(
        ErrorLog.job_id == job_id
    ).order_by(
        ErrorLog.timestamp.desc()
    ).limit(limit).all()


def get_all_jobs(db: Session, skip: int = 0, limit: int = 100):
    """Get all jobs with pagination"""
    return db.query(Job).order_by(Job.created_at.desc()).offset(skip).limit(limit).all()


def get_running_jobs(db: Session):
    """Get all running jobs"""
    return db.query(Job).filter(Job.status == "RUNNING").all()


def delete_job(db: Session, job_id: str) -> bool:
    """Delete a job"""
    job = get_job_by_id(db, job_id)
    if not job:
        return False

    # Delete associated error logs
    db.query(ErrorLog).filter(ErrorLog.job_id == job_id).delete()

    # Delete job
    db.delete(job)
    db.commit()
    return True
