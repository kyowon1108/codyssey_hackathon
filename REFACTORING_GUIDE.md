# Gaussian Splatting API 리팩토링 가이드

## 📁 새로운 프로젝트 구조

```
codyssey_hackathon/
├── app/                          ✅ 생성 완료
│   ├── __init__.py
│   ├── config.py                 ✅ 완료
│   │
│   ├── db/                       🔄 진행 중
│   │   ├── __init__.py
│   │   ├── database.py           # DB 연결 관리
│   │   ├── models.py             # SQLAlchemy 모델
│   │   └── crud.py               # CRUD 함수
│   │
│   ├── schemas/                  📝 작성 필요
│   │   ├── __init__.py
│   │   ├── job.py                # Job request/response 스키마
│   │   └── common.py             # 공통 스키마
│   │
│   ├── core/                     📝 작성 필요
│   │   ├── __init__.py
│   │   ├── colmap.py             # COLMAP 파이프라인
│   │   ├── gaussian_splatting.py # Gaussian Splatting 학습
│   │   ├── image_processing.py   # 이미지 검증/처리
│   │   └── pipeline.py           # 전체 파이프라인 오케스트레이션
│   │
│   ├── utils/                    📝 작성 필요
│   │   ├── __init__.py
│   │   ├── converter.py          # PLY ↔ Splat 변환
│   │   ├── filter.py             # Outlier 필터링
│   │   ├── logger.py             # 로깅 설정
│   │   └── system.py             # GPU 모니터링 등
│   │
│   ├── api/                      📝 작성 필요
│   │   ├── __init__.py
│   │   ├── dependencies.py       # FastAPI 의존성
│   │   ├── jobs.py               # Job API 라우터
│   │   └── viewer.py             # Viewer API 라우터
│   │
│   └── main.py                   📝 작성 필요 (핵심!)
│
├── templates/                    ✅ 디렉토리 생성 완료
│   └── viewer.html               # viewer_template.html 이동 예정
│
├── static/                       ✅ 디렉토리 생성 완료
│   └── (CSS, JS 파일 등)
│
├── tests/                        ✅ 디렉토리 생성 완료
│   ├── __init__.py
│   ├── test_api.py
│   ├── test_colmap.py
│   └── test_pipeline.py
│
├── scripts/                      ✅ 디렉토리 생성 완료
│   ├── setup.sh
│   └── migrate_db.py
│
├── logs/                         ✅ 디렉토리 생성 완료
├── data/                         (기존 유지)
├── gaussian-splatting/           (기존 유지)
│
├── .env                          📝 생성 필요
├── .gitignore                    🔄 업데이트 필요
├── requirements.txt              (기존 유지)
├── README.md                     🔄 업데이트 필요
└── pyproject.toml                📝 생성 권장 (선택)
```

---

## 🎯 단계별 마이그레이션 계획

### Phase 1: DB 모듈 분리 ✅ (이미 완료)
기존 `database.py`를 `app/db/` 아래로 분리

#### `app/db/models.py`
```python
from sqlalchemy import Column, String, DateTime, Integer, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class Job(Base):
    __tablename__ = "jobs"
    # ... (기존 database.py의 Job 모델)

class ErrorLog(Base):
    __tablename__ = "error_logs"
    # ... (기존 database.py의 ErrorLog 모델)
```

#### `app/db/database.py`
```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from app.config import settings

engine = create_engine(settings.DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db():
    from app.db.models import Base
    Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
```

#### `app/db/crud.py`
```python
from sqlalchemy.orm import Session
from app.db.models import Job, ErrorLog
from datetime import datetime
from typing import Optional

def create_job(db: Session, job_id: str, pub_key: str, **kwargs) -> Job:
    # ... (기존 database.py의 create_job 함수)
    pass

def get_job_by_id(db: Session, job_id: str) -> Optional[Job]:
    # ... (기존 database.py의 get_job_by_id 함수)
    pass

# ... 나머지 CRUD 함수들
```

---

### Phase 2: Utils 모듈 분리

#### `app/utils/converter.py`
기존 `convert_to_splat.py`의 내용을 여기로 이동

```python
from pathlib import Path

def ply_to_splat(ply_path: Path, splat_path: Path) -> None:
    """Convert PLY file to Splat format"""
    # ... (기존 convert_to_splat.py 로직)
    pass

def points3d_to_ply(points3d_txt: Path, ply_path: Path) -> None:
    """Convert COLMAP points3D.txt to PLY"""
    # ... (기존 main.py의 convert_points3d_to_ply 함수)
    pass
```

#### `app/utils/filter.py`
기존 `filter_outliers.py`를 모듈화

```python
from pathlib import Path
from typing import Tuple
import numpy as np

class OutlierFilter:
    def __init__(self, k_neighbors: int = 20, std_threshold: float = 2.0):
        self.k_neighbors = k_neighbors
        self.std_threshold = std_threshold

    def filter_ply(
        self,
        input_ply: Path,
        output_ply: Path,
        remove_small_clusters: bool = True
    ) -> Tuple[int, int, int]:
        """
        Filter outliers from PLY file

        Returns:
            (total_count, removed_count, remaining_count)
        """
        # ... (기존 filter_outliers.py 로직)
        pass
```

#### `app/utils/system.py`
GPU 모니터링 등 시스템 유틸리티

```python
import subprocess
from typing import Optional

def get_gpu_memory_usage() -> str:
    """Get GPU memory usage using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                used, total = line.split(',')
                used_mb = int(used.strip())
                total_mb = int(total.strip())
                percent = (used_mb / total_mb * 100) if total_mb > 0 else 0
                gpu_info.append(f"GPU {i}: {used_mb}MB / {total_mb}MB ({percent:.1f}%)")
            return " | ".join(gpu_info)
    except Exception as e:
        return f"GPU memory check failed: {str(e)}"
    return "N/A"
```

#### `app/utils/logger.py`
로깅 설정

```python
import logging
from app.config import settings

def setup_logger(name: str) -> logging.Logger:
    """Setup logger with consistent formatting"""
    logger = logging.getLogger(name)
    logger.setLevel(settings.LOG_LEVEL)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(settings.LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
```

---

### Phase 3: Core 비즈니스 로직 분리

#### `app/core/colmap.py`
COLMAP 파이프라인 클래스화

```python
import asyncio
from pathlib import Path
from typing import Optional
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class COLMAPPipeline:
    """COLMAP pipeline for Structure-from-Motion reconstruction"""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.database_path = job_dir / "colmap" / "database.db"
        self.images_path = job_dir / "upload" / "images"
        self.sparse_path = job_dir / "colmap" / "sparse"
        self.work_path = job_dir / "work"

    async def extract_features(self, log_file) -> None:
        """Extract SIFT features from images"""
        logger.info(f"Extracting features for job {self.job_dir.name}")

        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_path),
            "--ImageReader.camera_model", settings.COLMAP_CAMERA_MODEL,
            "--SiftExtraction.max_num_features", str(settings.COLMAP_MAX_FEATURES),
            "--FeatureExtraction.num_threads", str(settings.COLMAP_NUM_THREADS)
        ]

        await self._run_command(cmd, log_file)

    async def match_features(self, log_file) -> None:
        """Match features between images"""
        logger.info(f"Matching features for job {self.job_dir.name}")

        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--FeatureMatching.num_threads", str(settings.COLMAP_NUM_THREADS)
        ]

        await self._run_command(cmd, log_file)

    async def reconstruct(self, log_file) -> Path:
        """Perform sparse reconstruction (SfM)"""
        logger.info(f"Reconstructing sparse model for job {self.job_dir.name}")

        self.sparse_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_path),
            "--output_path", str(self.sparse_path)
        ]

        await self._run_command(cmd, log_file)

        model0_path = self.sparse_path / "0"
        if not model0_path.exists():
            raise RuntimeError("COLMAP reconstruction failed: no sparse model generated")

        return model0_path

    async def undistort_images(self, model_path: Path, log_file) -> Path:
        """Undistort images and prepare for Gaussian Splatting"""
        logger.info(f"Undistorting images for job {self.job_dir.name}")

        self.work_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(self.images_path),
            "--input_path", str(model_path),
            "--output_path", str(self.work_path),
            "--output_type", "COLMAP"
        ]

        await self._run_command(cmd, log_file)

        # GS expects sparse model in sparse/0/ subdirectory
        sparse_0_dir = self.work_path / "sparse" / "0"
        sparse_0_dir.mkdir(parents=True, exist_ok=True)

        # Move sparse files
        import shutil
        sparse_dir = self.work_path / "sparse"
        for item in sparse_dir.iterdir():
            if item.is_file():
                shutil.move(str(item), str(sparse_0_dir / item.name))

        return self.work_path

    async def _run_command(self, cmd: list, log_file):
        """Run subprocess command with logging"""
        from app.core.pipeline import run_command
        await run_command(cmd, log_file)
```

#### `app/core/gaussian_splatting.py`
Gaussian Splatting 학습 클래스

```python
import asyncio
from pathlib import Path
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

class GaussianSplattingTrainer:
    """Gaussian Splatting training pipeline"""

    def __init__(self, work_dir: Path, output_dir: Path):
        self.work_dir = work_dir
        self.output_dir = output_dir

    async def train(self, log_file, iterations: int = None) -> Path:
        """Train Gaussian Splatting model"""
        iterations = iterations or settings.TRAINING_ITERATIONS

        logger.info(f"Training Gaussian Splatting for {iterations} iterations")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        gs_script = settings.GAUSSIAN_SPLATTING_DIR / "train.py"

        import os
        env = os.environ.copy()
        torch_lib = settings.CONDA_PYTHON.parent.parent / "lib" / "python3.9" / "site-packages" / "torch" / "lib"
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        env["PYTHONPATH"] = str(settings.GAUSSIAN_SPLATTING_DIR)

        cmd = [
            str(settings.CONDA_PYTHON),
            str(gs_script),
            "-s", str(self.work_dir),
            "-m", str(self.output_dir),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--densify_until_iter", str(settings.DENSIFY_UNTIL_ITER),
            "--densification_interval", str(settings.DENSIFICATION_INTERVAL),
            "--opacity_reset_interval", str(settings.OPACITY_RESET_INTERVAL),
            "--resolution", "1"
        ]

        from app.core.pipeline import run_command
        await run_command(cmd, log_file, env=env, monitor_gpu=True)

        iteration_dir = self.output_dir / "point_cloud" / f"iteration_{iterations}"
        return iteration_dir
```

#### `app/core/pipeline.py`
전체 파이프라인 오케스트레이션 + run_command 함수

```python
import asyncio
from pathlib import Path
from typing import Optional
from app.config import settings
from app.core.colmap import COLMAPPipeline
from app.core.gaussian_splatting import GaussianSplattingTrainer
from app.utils.system import get_gpu_memory_usage
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

async def run_command(cmd: list, log_file, cwd: Path = None, env: dict = None, monitor_gpu: bool = False):
    """Run subprocess command with logging and GPU monitoring"""
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=(str(cwd) if cwd else None),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    gpu_check_counter = 0

    while True:
        try:
            chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=1.0)
            if not chunk:
                break
            text = chunk.decode(errors="ignore")
            log_file.write(text)
            log_file.flush()

            if monitor_gpu:
                gpu_check_counter += 1
                if gpu_check_counter >= settings.GPU_CHECK_INTERVAL:
                    gpu_check_counter = 0
                    gpu_mem = get_gpu_memory_usage()
                    log_file.write(f"\n[GPU Memory] {gpu_mem}\n")
                    log_file.flush()

        except asyncio.TimeoutError:
            if process.returncode is not None:
                break
            continue

    exit_code = await process.wait()
    if exit_code != 0:
        log_file.write(f"[ERROR] Command {' '.join(cmd)} exited with code {exit_code}\n")
        log_file.flush()
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (exit code: {exit_code})")


class ReconstructionPipeline:
    """Complete reconstruction pipeline orchestrator"""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.colmap = COLMAPPipeline(job_dir)

        gs_output_dir = job_dir / "gs_output"
        work_dir = job_dir / "work"
        self.gs_trainer = GaussianSplattingTrainer(work_dir, gs_output_dir)

    async def run(self, log_file) -> None:
        """Execute complete pipeline"""
        logger.info(f"Starting reconstruction pipeline for job {self.job_dir.name}")

        # COLMAP pipeline
        log_file.write(">> [1/6] Feature extraction\n")
        await self.colmap.extract_features(log_file)

        log_file.write(">> [2/6] Feature matching\n")
        await self.colmap.match_features(log_file)

        log_file.write(">> [3/6] Sparse reconstruction\n")
        model_path = await self.colmap.reconstruct(log_file)

        log_file.write(">> [4/6] Image undistortion\n")
        work_path = await self.colmap.undistort_images(model_path, log_file)

        # Gaussian Splatting
        log_file.write(">> [5/6] Gaussian Splatting training\n")
        iteration_dir = await self.gs_trainer.train(log_file)

        # Post-processing
        log_file.write(">> [6/6] Post-processing\n")
        await self._post_process(iteration_dir, log_file)

        logger.info(f"Pipeline completed for job {self.job_dir.name}")

    async def _post_process(self, iteration_dir: Path, log_file):
        """Post-processing: filtering and conversion"""
        from app.utils.filter import OutlierFilter
        from app.utils.converter import ply_to_splat
        import subprocess

        ply_file = iteration_dir / "point_cloud.ply"
        filtered_ply = iteration_dir / "point_cloud_filtered.ply"
        splat_file = iteration_dir / "scene.splat"

        if ply_file.exists() and not filtered_ply.exists():
            log_file.write(">> Applying outlier filtering...\n")

            filter_cmd = [
                str(settings.CONDA_PYTHON),
                str(settings.BASE_DIR / "filter_outliers.py"),
                str(ply_file),
                str(filtered_ply),
                "--k_neighbors", str(settings.OUTLIER_K_NEIGHBORS),
                "--std_threshold", str(settings.OUTLIER_STD_THRESHOLD),
                "--remove_small_clusters",
                "--min_cluster_ratio", str(settings.OUTLIER_MIN_CLUSTER_RATIO)
            ]
            result = subprocess.run(filter_cmd, capture_output=True, text=True)
            log_file.write(result.stdout)
            log_file.write(result.stderr)

        # Convert to splat
        ply_for_conversion = filtered_ply if filtered_ply.exists() else ply_file
        if ply_for_conversion.exists() and not splat_file.exists():
            log_file.write(">> Converting to splat format...\n")
            convert_cmd = [
                "python",
                str(settings.BASE_DIR / "convert_to_splat.py"),
                str(ply_for_conversion),
                str(splat_file)
            ]
            subprocess.run(convert_cmd, check=True)
            log_file.write(">> Conversion complete!\n")
```

---

### Phase 4: Schemas 작성

#### `app/schemas/job.py`
Pydantic 스키마

```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional

class JobCreateRequest(BaseModel):
    """Request schema for creating a job"""
    original_resolution: bool = Field(default=False, description="Use original image resolution")

class JobCreateResponse(BaseModel):
    """Response schema after creating a job"""
    job_id: str
    pub_key: str
    original_resolution: bool

class JobStatusResponse(BaseModel):
    """Response schema for job status"""
    job_id: str
    status: str
    log_tail: list[str] = []
    created_at: Optional[str] = None
    processing_time_seconds: Optional[float] = None
    image_count: Optional[int] = None
    gaussian_count: Optional[int] = None
    file_size_mb: Optional[float] = None
    viewer_url: Optional[str] = None
    error: Optional[str] = None
    error_stage: Optional[str] = None
```

---

### Phase 5: API 라우터 분리

#### `app/api/dependencies.py`
공통 의존성

```python
from fastapi import Depends
from sqlalchemy.orm import Session
from app.db.database import get_db

# 이미 get_db가 정의되어 있으므로 여기서는 재사용
# 필요한 경우 추가 의존성 정의
```

#### `app/api/jobs.py`
Job API 라우터

```python
import uuid
import random
import asyncio
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from fastapi.responses import JSONResponse, FileResponse
from sqlalchemy.orm import Session
from PIL import Image
import io

from app.db.database import get_db
from app.db import crud
from app.schemas.job import JobCreateRequest, JobCreateResponse, JobStatusResponse
from app.config import settings
from app.core.pipeline import ReconstructionPipeline
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

router = APIRouter(prefix="/recon/jobs", tags=["jobs"])

# Semaphore for concurrent job control
sem = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)


def validate_image_file(file: UploadFile):
    """Validate uploaded image file"""
    file_ext = Path(file.filename).suffix
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type: {file_ext}")

    try:
        content = file.file.read()
        img = Image.open(io.BytesIO(content))
        img.verify()
        file.file.seek(0)

        if img.width < settings.MIN_IMAGE_SIZE or img.height < settings.MIN_IMAGE_SIZE:
            raise HTTPException(400, f"Image too small: {img.width}x{img.height}")

        logger.info(f"Validated image: {file.filename} ({img.width}x{img.height})")
    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {file.filename}. Error: {str(e)}")


async def process_job(job_id: str):
    """Background job processor"""
    async with sem:
        db = next(get_db())
        try:
            job = crud.get_job_by_id(db, job_id)
            if not job:
                logger.error(f"Job {job_id} not found")
                return

            crud.update_job_status(db, job_id, "RUNNING")

            job_dir = settings.DATA_DIR / job_id
            log_path = job_dir / "run.log"

            with open(log_path, 'a', encoding='utf-8', buffering=1) as log_file:
                log_file.write(f"=== Job {job_id} started ===\n")

                pipeline = ReconstructionPipeline(job_dir)
                await pipeline.run(log_file)

                crud.update_job_status(db, job_id, "DONE")
                log_file.write(f"=== Job {job_id} completed ===\n")

        except Exception as e:
            logger.error(f"Job {job_id} failed: {str(e)}")
            import traceback
            crud.log_error(db, job_id, "general", type(e).__name__, str(e), traceback.format_exc())
            crud.update_job_status(db, job_id, "FAILED", error_message=str(e))

        finally:
            db.close()


@router.post("/", response_model=JobCreateResponse)
async def create_job(
    files: list[UploadFile] = File(...),
    original_resolution: bool = False,
    db: Session = Depends(get_db)
):
    """Create a new reconstruction job"""
    # Validate images
    for file in files:
        validate_image_file(file)

    # Generate IDs
    job_id = uuid.uuid4().hex[:8]
    pub_key = ''.join(random.choice("0123456789") for _ in range(10))

    # Create job in database
    try:
        job = crud.create_job(
            db,
            job_id=job_id,
            pub_key=pub_key,
            image_count=len(files),
            original_resolution=original_resolution
        )
    except Exception as e:
        logger.error(f"Failed to create job: {e}")
        raise HTTPException(500, "Failed to create job")

    # Create directories and save images
    job_dir = settings.DATA_DIR / job_id
    images_dir = job_dir / "upload" / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "colmap").mkdir(parents=True, exist_ok=True)

    # Save images
    for file in files:
        content = await file.read()
        img_path = images_dir / file.filename

        if original_resolution:
            with open(img_path, 'wb') as f:
                f.write(content)
        else:
            # Resize images
            try:
                img = Image.open(io.BytesIO(content))
                width, height = img.size

                if width > settings.MAX_IMAGE_SIZE or height > settings.MAX_IMAGE_SIZE:
                    ratio = min(settings.MAX_IMAGE_SIZE / width, settings.MAX_IMAGE_SIZE / height)
                    new_width = int(width * ratio)
                    new_height = int(height * ratio)
                    img = img.resize((new_width, new_height), Image.LANCZOS)

                img.save(img_path, quality=95)
            except Exception as e:
                logger.warning(f"Failed to resize image {file.filename}: {e}. Saving original.")
                with open(img_path, 'wb') as f:
                    f.write(content)

    # Start background processing
    asyncio.create_task(process_job(job_id))

    return JobCreateResponse(
        job_id=job_id,
        pub_key=pub_key,
        original_resolution=original_resolution
    )


@router.get("/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    """Get job status"""
    job = crud.get_job_by_id(db, job_id)

    if not job:
        raise HTTPException(404, "Job not found")

    # Read log tail
    log_path = settings.DATA_DIR / job_id / "run.log"
    log_tail = []
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                log_tail = lines[-10:] if lines else []
        except Exception as e:
            log_tail = [f"(log read error: {e})"]

    response = JobStatusResponse(
        job_id=job.job_id,
        status=job.status,
        log_tail=log_tail,
        created_at=job.created_at.isoformat() if job.created_at else None,
        processing_time_seconds=job.processing_time_seconds,
        image_count=job.image_count,
        gaussian_count=job.gaussian_count,
        file_size_mb=job.file_size_mb
    )

    if job.status == "DONE":
        response.viewer_url = f"/v/{job.pub_key}"
    elif job.status == "FAILED":
        response.error = job.error_message
        response.error_stage = job.error_stage

    return response


@router.get("/pub/{pub_key}/cloud.ply")
async def download_ply(pub_key: str, db: Session = Depends(get_db)):
    """Download PLY file"""
    job = crud.get_job_by_pub_key(db, pub_key)

    if not job or job.status != "DONE":
        raise HTTPException(404, "File not found")

    gs_ply = settings.DATA_DIR / job.job_id / "gs_output" / "point_cloud" / "iteration_10000" / "point_cloud.ply"
    if gs_ply.exists():
        return FileResponse(str(gs_ply), media_type="application/octet-stream", filename="cloud.ply")

    raise HTTPException(404, "cloud.ply not found")


@router.get("/pub/{pub_key}/scene.splat")
async def download_splat(pub_key: str, db: Session = Depends(get_db)):
    """Download Splat file"""
    job = crud.get_job_by_pub_key(db, pub_key)

    if not job or job.status != "DONE":
        raise HTTPException(404, "File not found")

    splat_file = settings.DATA_DIR / job.job_id / "gs_output" / "point_cloud" / "iteration_10000" / "scene.splat"
    if splat_file.exists():
        from fastapi import Response
        with open(splat_file, 'rb') as f:
            content = f.read()
        return Response(
            content=content,
            media_type="application/octet-stream",
            headers={
                "X-Iteration": "10000",
                "Content-Disposition": "attachment; filename=scene_10000.splat"
            }
        )

    raise HTTPException(404, "scene.splat not found")
```

#### `app/api/viewer.py`
Viewer API 라우터

```python
from fastapi import APIRouter, HTTPException, Request, Depends
from fastapi.responses import HTMLResponse
from sqlalchemy.orm import Session
from pathlib import Path

from app.db.database import get_db
from app.db import crud
from app.config import settings

router = APIRouter(prefix="/v", tags=["viewer"])


@router.get("/{pub_key}", response_class=HTMLResponse)
async def view_result(pub_key: str, request: Request, db: Session = Depends(get_db)):
    """View 3D result in web viewer"""
    job = crud.get_job_by_pub_key(db, pub_key)

    if not job:
        raise HTTPException(404, "Invalid viewer key")

    if job.status != "DONE":
        return HTMLResponse(
            content="<html><body><h3>Result is not ready yet. Please check again later.</h3></body></html>"
        )

    # Load viewer template
    viewer_path = settings.TEMPLATES_DIR / "viewer.html"
    with open(viewer_path, 'r') as f:
        html_content = f.read()

    # Replace splat URL
    splat_url = f"/recon/pub/{pub_key}/scene.splat"
    html_content = html_content.replace('SPLAT_URL_PLACEHOLDER', splat_url)

    return HTMLResponse(content=html_content, status_code=200)
```

---

### Phase 6: Main App 재구성

#### `app/main.py`
핵심! 모든 것을 연결하는 메인 앱

```python
"""
Gaussian Splatting 3D Reconstruction API
Main application entry point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.db.database import init_db
from app.api import jobs, viewer
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup"""
    logger.info("Starting Gaussian Splatting API...")

    # Initialize database
    init_db()
    logger.info("Database initialized")

    # Ensure directories exist
    settings.ensure_dirs()
    logger.info("Directories ensured")

    logger.info(f"Application started successfully on {settings.APP_VERSION}")


# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down Gaussian Splatting API...")


# Include routers
app.include_router(jobs.router)
app.include_router(viewer.router)


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "ok",
        "version": settings.APP_VERSION
    }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Gaussian Splatting 3D Reconstruction API",
        "version": settings.APP_VERSION,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

---

### Phase 7: 실행 방법

#### 기존 방식 (현재)
```bash
python main.py
```

#### 새로운 방식
```bash
# 방법 1: 직접 실행
python -m app.main

# 방법 2: uvicorn 사용 (권장)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 방법 3: 프로덕션 (workers)
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🔄 마이그레이션 체크리스트

### ✅ 완료
- [x] 디렉토리 구조 생성
- [x] `app/config.py` 작성
- [x] `database.py` 생성 (루트에 이미 존재)

### 📝 작업 필요
- [ ] `database.py` → `app/db/` 분리
- [ ] `filter_outliers.py` → `app/utils/filter.py` 모듈화
- [ ] `convert_to_splat.py` → `app/utils/converter.py` 모듈화
- [ ] `main.py` → `app/main.py` + `app/api/` + `app/core/` 분리
- [ ] `viewer_template.html` → `templates/viewer.html` 이동
- [ ] `.gitignore` 업데이트
- [ ] `README.md` 업데이트

---

## 🎯 실행 전 체크리스트

1. **의존성 설치**
   ```bash
   pip install -r requirements.txt
   ```

2. **데이터베이스 초기화**
   ```bash
   python -c "from app.db.database import init_db; init_db()"
   ```

3. **템플릿 파일 이동**
   ```bash
   mv viewer_template.html templates/viewer.html
   ```

4. **환경 변수 설정** (선택)
   ```bash
   # .env 파일 생성
   echo "MAX_CONCURRENT_JOBS=2" > .env
   echo "TRAINING_ITERATIONS=10000" >> .env
   ```

5. **서버 실행**
   ```bash
   uvicorn app.main:app --reload
   ```

---

## 📌 주의사항

1. **기존 main.py 백업**: 이미 `main_backup.py`로 백업됨
2. **데이터 마이그레이션**: 기존 `gaussian_splatting.db`가 있다면 그대로 사용 가능
3. **점진적 전환**: 한 번에 모두 바꾸기보다는 모듈별로 테스트하며 전환
4. **테스트**: 각 API 엔드포인트 테스트 필수

---

## 🚀 다음 단계

1. **현재까지 완료된 작업 Git commit**
2. **핵심 파일부터 마이그레이션 시작**
3. **테스트 작성**
4. **문서화 업데이트**

---

**작성일**: 2025-10-13
**버전**: 2.0 Refactoring Plan
**작성자**: Claude Code
