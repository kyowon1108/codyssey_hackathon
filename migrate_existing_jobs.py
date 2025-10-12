#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Migrate existing job data to database
"""
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from app.db.database import SessionLocal, init_db
from app.db import crud
from app.config import settings


def get_job_info(job_dir: Path):
    """Extract job information from directory"""
    job_id = job_dir.name
    
    # Read public key
    key_file = job_dir / "key.txt"
    if not key_file.exists():
        print(f"  Warning: No key.txt found for {job_id}")
        return None
    
    pub_key = key_file.read_text().strip()
    
    # Count images
    images_dir = job_dir / "upload" / "images"
    image_count = 0
    if images_dir.exists():
        image_count = len(list(images_dir.glob("*.*")))
    
    # Check if completed
    output_dir = job_dir / "output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}"
    gs_output_dir = job_dir / "gs_output" / "point_cloud" / f"iteration_{settings.TRAINING_ITERATIONS}"
    
    status = "COMPLETED"
    gaussian_count = None
    
    # Try new structure first
    ply_file = output_dir / "point_cloud_filtered.ply"
    if not ply_file.exists():
        ply_file = output_dir / "point_cloud.ply"
    
    # Try old structure
    if not ply_file.exists():
        ply_file = gs_output_dir / "point_cloud_filtered.ply"
    if not ply_file.exists():
        ply_file = gs_output_dir / "point_cloud.ply"
    
    if ply_file.exists():
        # Count Gaussians (read as binary to handle encoding issues)
        try:
            with open(ply_file, 'rb') as f:
                for line in f:
                    line_str = line.decode('utf-8', errors='ignore')
                    if line_str.startswith("element vertex"):
                        gaussian_count = int(line_str.split()[-1])
                        break
        except Exception as e:
            print(f"  Warning: Error reading PLY: {e}")
    else:
        status = "UNKNOWN"
    
    created_at = datetime.fromtimestamp(job_dir.stat().st_ctime)
    
    return {
        "job_id": job_id,
        "pub_key": pub_key,
        "status": status,
        "image_count": image_count,
        "gaussian_count": gaussian_count,
        "created_at": created_at
    }


def migrate_jobs():
    """Migrate all existing jobs to database"""
    print("=" * 60)
    print("Migrating existing jobs to database")
    print("=" * 60)
    
    init_db()
    print(f"✓ Database initialized: {settings.DATABASE_URL}\n")
    
    jobs_dir = settings.DATA_DIR
    if not jobs_dir.exists():
        print(f"✗ Jobs directory not found: {jobs_dir}")
        return
    
    job_dirs = [d for d in jobs_dir.iterdir() if d.is_dir()]
    print(f"Found {len(job_dirs)} job directories\n")
    
    db = SessionLocal()
    migrated = 0
    skipped = 0
    errors = 0
    
    try:
        for job_dir in sorted(job_dirs):
            job_id = job_dir.name
            
            existing = crud.get_job_by_id(db, job_id)
            if existing:
                print(f"⊘ {job_id}: Already in database")
                skipped += 1
                continue
            
            job_info = get_job_info(job_dir)
            if not job_info:
                print(f"✗ {job_id}: Failed to extract info")
                errors += 1
                continue
            
            try:
                crud.create_job(
                    db=db,
                    job_id=job_info["job_id"],
                    pub_key=job_info["pub_key"],
                    image_count=job_info["image_count"],
                    original_resolution=False
                )
                
                crud.update_job_status(
                    db=db,
                    job_id=job_info["job_id"],
                    status=job_info["status"]
                )
                
                if job_info["gaussian_count"]:
                    crud.update_job_results(
                        db=db,
                        job_id=job_info["job_id"],
                        gaussian_count=job_info["gaussian_count"]
                    )
                
                db.commit()
                
                status_str = f"{job_info['status']}"
                if job_info['gaussian_count']:
                    status_str += f" ({job_info['gaussian_count']:,} Gaussians)"
                
                print(f"✓ {job_id}: {status_str}")
                migrated += 1
                
            except Exception as e:
                print(f"✗ {job_id}: Database error - {e}")
                errors += 1
                db.rollback()
    
    finally:
        db.close()
    
    print("\n" + "=" * 60)
    print("Migration Summary")
    print("=" * 60)
    print(f"Total jobs found:     {len(job_dirs)}")
    print(f"✓ Migrated:           {migrated}")
    print(f"⊘ Already in DB:      {skipped}")
    print(f"✗ Errors:             {errors}")
    print("=" * 60)
    
    if migrated > 0:
        print(f"\n✓ Successfully migrated {migrated} jobs to database!")
    else:
        print("\n⊘ No new jobs to migrate.")


if __name__ == "__main__":
    migrate_jobs()
