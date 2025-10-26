# 3DGS Server MVP Refactoring Guide (수정판)

## 🎯 목표
해커톤 MVP에 맞춰 InstaRecon3D (3DGS Server)를 간소화하고 성능을 최적화합니다.

---

## 📋 Git 작업 흐름

```bash
# 1. 새 브랜치 생성
git checkout -b feature/mvp-refactor

# 2. 중간 상태 커밋 (작업 단계마다)
git add .
git commit -m "refactor: [작업 내용]"

# 3. 최종 완료 후
git push origin feature/mvp-refactor
```

---

## ⚠️ 중요 전제 조건

**Backend API 서버가 아직 구현되지 않았으므로 다음 기능은 후순위로 미룹니다:**

1. ❌ **Callback URL** → Backend 구현 후 추가
2. ❌ **S3 Image Download** → S3 크레딧 확정 후 추가  
3. ❌ **Job ID 외부 입력** → Backend 구현 후 추가

**현재 작업 범위:**
- ✅ 불필요한 기능 제거 (Preflight, Evaluation, Outlier Filtering 등)
- ✅ DB 간소화 (Job 테이블만 관리)
- ✅ 이미지 리사이징 최적화 (1280px)
- ✅ 성능 개선

---

## 🔧 작업 항목 상세

### 1. Preflight Check 제거 (매 작업마다) ⭐⭐⭐

**파일**: `app/api/jobs.py`, `app/main.py`

#### 🤔 왜 제거하나요?

**현재 상황**:
```python
# app/api/jobs.py - process_job() 함수
async def process_job(job_id: str, ...):
    # 매 작업마다 실행됨
    try:
        update_job_status(db, job_id, "PROCESSING", "PREFLIGHT", 5)
        preflight_result = run_preflight_check()  # ← 매번 체크
        if not preflight_result.passed:
            raise Exception("Preflight failed")
    except Exception as e:
        ...
```

**문제점**:
1. **불필요한 반복**: 매 작업마다 Python 버전, CUDA, COLMAP 등을 체크
2. **시간 낭비**: 1-2초씩 소요 (환경은 한 번 검증하면 충분)
3. **개발 환경은 이미 검증됨**: 해커톤 기간 동안 환경 변경 없음

**해결 방법**:
- 서버 **시작 시 1회만** 환경 체크
- 실패 시 서버 자체가 시작 안 됨 → 더 안전함

#### 🔧 작업

**1. `app/api/jobs.py`에서 Preflight Check 제거:**

```python
# app/api/jobs.py (Line ~354-368)
# 제거할 부분:
# try:
#     update_job_status(db, job_id, "PROCESSING", "PREFLIGHT", 5)
#     preflight_result = run_preflight_check()
#     if not preflight_result.passed:
#         raise Exception(f"Preflight check failed: {preflight_result.get_summary()}")
# except Exception as e:
#     handle_job_error(db, job_id, "PREFLIGHT", str(e), traceback.format_exc())
#     return

# 대신 바로 COLMAP으로 진행:
update_job_status(db, job_id, "PROCESSING", "COLMAP_FEAT", 15)
```

**2. `app/main.py`에 서버 시작 시 검사 추가:**

```python
# app/main.py
from app.utils.preflight import run_preflight_check
from app.utils.logger import logger

@app.on_event("startup")
async def startup_event():
    """서버 시작 시 환경 검증 (1회만)"""
    logger.info("=" * 50)
    logger.info("Running preflight check on server startup...")
    logger.info("=" * 50)
    
    preflight_result = run_preflight_check()
    
    if not preflight_result.passed:
        logger.error("❌ Preflight check failed!")
        logger.error(f"\n{preflight_result.get_summary()}")
        raise RuntimeError(
            "Server environment is not ready. "
            "Please fix the issues above before starting the server."
        )
    
    logger.info("✅ Preflight check passed! Server is ready.")
    logger.info("=" * 50)
```

#### 📊 효과
- **절약 시간**: 작업당 1-2초
- **안전성**: 서버 시작 실패 시 즉시 감지
- **로그**: 작업 로그가 깔끔해짐 (Preflight 단계 제거)

**커밋**:
```bash
git add app/api/jobs.py app/main.py
git commit -m "refactor: move preflight check to server startup (1-2s saved per job)"
```

---

### 2. Model Evaluation 제거 ⭐⭐⭐

**파일**: `app/api/jobs.py`, `app/core/gaussian_splatting.py`, `app/db/models.py`

#### 🤔 왜 제거하나요?

**현재 상황**:
```python
# app/api/jobs.py - process_job() 함수
try:
    update_job_status(db, job_id, "PROCESSING", "EVALUATION", 85)
    evaluate(job_dir, settings.TRAINING_ITERATIONS)  # ← 30-60초 소요
    
    # results.json 파일 읽기
    results = load_evaluation_results(job_dir)
    psnr = results['PSNR']
    ssim = results['SSIM']
    lpips = results['LPIPS']
    
    # DB에 저장
    crud.update_job(db, job_id, {
        'psnr': psnr,
        'ssim': ssim,
        'lpips': lpips
    })
except Exception as e:
    ...
```

**무엇을 하는가?**:
1. Test set 이미지 렌더링 (20% 이미지)
2. Ground truth와 비교
3. PSNR, SSIM, LPIPS 계산

**문제점**:
1. **DB 설계에서 이미 제거 확정**: psnr, ssim, lpips 컬럼 없음
2. **30-60초 소요**: 전체 처리 시간의 5-10%
3. **사용자는 사용 안 함**: 3D 뷰어로 직접 보면 됨
4. **기술적 메트릭**: 일반 사용자에게 의미 없음 (PSNR 22dB가 뭔지 모름)

**해결 방법**:
- Evaluation 단계 완전 제거
- DB 메트릭 컬럼 제거
- Train/Test Split도 불필요 (Evaluation 안 하므로)

#### 🔧 작업

**1. `app/api/jobs.py`에서 Evaluation 제거:**

```python
# app/api/jobs.py (Line ~446-451)
# 제거할 부분:
# try:
#     update_job_status(db, job_id, "PROCESSING", "EVALUATION", 85)
#     evaluate(job_dir, settings.TRAINING_ITERATIONS)
#     
#     # Load evaluation results
#     results_file = Path(job_dir) / "output" / "results.json"
#     if results_file.exists():
#         with open(results_file, 'r') as f:
#             results = json.load(f)
#             psnr = results.get('ours_7000', {}).get('PSNR')
#             ssim = results.get('ours_7000', {}).get('SSIM')
#             lpips = results.get('ours_7000', {}).get('LPIPS')
# except Exception as e:
#     handle_job_error(db, job_id, "EVALUATION", str(e), traceback.format_exc())
#     return

# 대신 바로 post-processing으로:
update_job_status(db, job_id, "PROCESSING", "EXPORT_PLY", 95)
```

**2. `app/db/models.py`에서 메트릭 컬럼 제거:**

```python
# app/db/models.py - Job 모델
class Job(Base):
    __tablename__ = "jobs"
    
    job_id = Column(String, primary_key=True, index=True)
    pub_key = Column(String, unique=True, nullable=False, index=True)
    status = Column(String, nullable=False)
    step = Column(String, nullable=True)
    progress = Column(Integer, default=0)
    image_count = Column(Integer, nullable=False)
    iterations = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    error_message = Column(Text, nullable=True)
    
    # 제거할 컬럼들 (주석 처리 또는 삭제):
    # psnr = Column(Float, nullable=True)        # ← 제거
    # ssim = Column(Float, nullable=True)        # ← 제거
    # lpips = Column(Float, nullable=True)       # ← 제거
    # gaussian_count = Column(Integer, nullable=True)  # ← 제거
```

**3. DB 마이그레이션 스크립트:**

```python
# migrations/remove_metrics.py
"""
DB 마이그레이션: 평가 메트릭 컬럼 제거
- psnr, ssim, lpips, gaussian_count 제거
"""
import sqlite3
from pathlib import Path

def migrate():
    db_path = Path(__file__).parent.parent / "data" / "jobs.db"
    
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    print("📊 Starting migration: remove evaluation metrics...")
    
    # SQLite는 ALTER TABLE DROP COLUMN을 지원하지 않으므로
    # 테이블 재생성 필요
    
    # 1. 백업 테이블 생성 (기존 데이터 보존)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS jobs_backup AS 
        SELECT * FROM jobs
    """)
    print("✓ Backup created")
    
    # 2. 기존 테이블 삭제
    cursor.execute("DROP TABLE IF EXISTS jobs")
    print("✓ Old table dropped")
    
    # 3. 새 테이블 생성 (메트릭 컬럼 제외)
    cursor.execute("""
        CREATE TABLE jobs (
            job_id TEXT PRIMARY KEY,
            pub_key TEXT UNIQUE NOT NULL,
            status TEXT NOT NULL,
            step TEXT,
            progress INTEGER DEFAULT 0,
            image_count INTEGER NOT NULL,
            iterations INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            error_message TEXT
        )
    """)
    print("✓ New table created")
    
    # 4. 데이터 복사 (메트릭 제외)
    cursor.execute("""
        INSERT INTO jobs 
        SELECT job_id, pub_key, status, step, progress, image_count, 
               iterations, created_at, completed_at, error_message
        FROM jobs_backup
    """)
    
    # 5. 백업 테이블 삭제
    cursor.execute("DROP TABLE jobs_backup")
    
    conn.commit()
    conn.close()
    
    print("✅ Migration completed!")
    print("   - Removed columns: psnr, ssim, lpips, gaussian_count")

if __name__ == "__main__":
    migrate()
```

**실행**:
```bash
python migrations/remove_metrics.py
```

#### 📊 효과
- **절약 시간**: 작업당 30-60초 (전체의 5-10%)
- **DB 간소화**: 4개 컬럼 제거
- **코드 간소화**: Evaluation 관련 코드 제거

**커밋**:
```bash
python migrations/remove_metrics.py
git add app/api/jobs.py app/db/models.py migrations/
git commit -m "refactor: remove evaluation (psnr/ssim/lpips) - saves 30-60s per job"
```

---

### 3. Train/Test Split 제거 ⭐⭐

**파일**: `app/api/jobs.py`, `app/core/colmap.py`

#### 🤔 왜 제거하나요?

**현재 상황**:
```python
# app/api/jobs.py - process_job() 함수
try:
    # Train/Test 분할 (80% train, 20% test)
    create_train_test_split(work_dir, train_ratio=0.8)
except Exception as e:
    ...
```

**무엇을 하는가?**:
- 이미지를 80% train, 20% test로 분할
- `work/train/`, `work/test/` 디렉토리에 심볼릭 링크 생성
- Test set으로 Evaluation 수행

**문제점**:
1. **Evaluation 제거했으므로 불필요**: Test set 사용 안 함
2. **5-10초 소요**: 심볼릭 링크 생성 시간
3. **디스크 공간 낭비**: train/test 디렉토리 생성

**해결 방법**:
- Train/Test Split 완전 제거
- 모든 이미지를 학습에 사용

#### 🔧 작업

```python
# app/api/jobs.py (Line ~407-410)
# 제거할 부분:
# try:
#     logger.info("Creating train/test split...")
#     create_train_test_split(work_dir, train_ratio=0.8)
# except Exception as e:
#     logger.error(f"Train/test split failed: {e}")
#     handle_job_error(db, job_id, "TRAIN_TEST_SPLIT", str(e), traceback.format_exc())
#     return

# 삭제 후 바로 다음 단계로 진행 (Validation 또는 Training)
```

#### 📊 효과
- **절약 시간**: 작업당 5-10초
- **디스크 절약**: train/test 디렉토리 불필요
- **코드 간소화**: 분할 로직 제거

**커밋**:
```bash
git add app/api/jobs.py
git commit -m "refactor: remove train/test split (not needed without evaluation)"
```

---

### 4. Outlier Filtering 제거 ⭐⭐

**파일**: `app/core/gaussian_splatting.py`, `app/utils/outlier_filter.py`

#### 🤔 왜 제거하나요?

**현재 상황**:
```python
# app/core/gaussian_splatting.py - post_process() 함수
def post_process(output_dir: str, iterations: int):
    ply_dir = Path(output_dir) / "point_cloud" / f"iteration_{iterations}"
    ply_file = ply_dir / "point_cloud.ply"
    
    # 1. Outlier filtering (10-20초 소요)
    filtered_file = ply_dir / "point_cloud_filtered.ply"
    filter_outliers(ply_file, filtered_file)  # ← K-NN + DBSCAN
    
    # 2. GZIP compression
    compress_ply(ply_file)
    compress_ply(filtered_file)
```

**무엇을 하는가?**:
```python
# app/utils/outlier_filter.py
def filter_outliers(input_file, output_file):
    # 1. K-NN Outlier Detection
    #    - 각 Gaussian의 20개 이웃까지 거리 계산
    #    - 평균 거리가 2.0 std 이상이면 outlier
    
    # 2. DBSCAN Clustering
    #    - 작은 클러스터 제거 (전체의 1% 미만)
    #    - 고립된 노이즈 제거
    
    # 결과: 일반적으로 5~10% Gaussian 제거
```

**문제점**:
1. **10-20초 소요**: 복잡한 알고리즘 (K-NN + DBSCAN)
2. **효과 미미**: Gaussian Splatting 자체가 이미 노이즈에 강함
3. **사용자 체감 어려움**: 5-10% 제거해도 뷰어에서 차이 거의 없음
4. **MVP 불필요**: 품질 문제 발생 시 추후 추가 가능

**Gaussian Splatting이 노이즈에 강한 이유**:
- Opacity 학습으로 자동 필터링
- Densification/Pruning 과정에서 노이즈 제거
- Adaptive density control

**해결 방법**:
- Outlier filtering 완전 제거
- GZIP compression만 수행

#### 🔧 작업

```python
# app/core/gaussian_splatting.py - post_process() 수정
def post_process(output_dir: str, iterations: int):
    """Post-processing: GZIP compression only (outlier filtering removed)"""
    ply_dir = Path(output_dir) / "point_cloud" / f"iteration_{iterations}"
    ply_file = ply_dir / "point_cloud.ply"
    
    if not ply_file.exists():
        raise FileNotFoundError(f"PLY file not found: {ply_file}")
    
    logger.info("Starting post-processing (compression only)...")
    
    # Outlier filtering 제거 (기존 코드 주석 또는 삭제)
    # filtered_file = ply_dir / "point_cloud_filtered.ply"
    # filter_outliers(ply_file, filtered_file)
    
    # GZIP compression만 수행
    compress_ply(ply_file)
    
    logger.info("✓ Post-processing completed")
```

**PLY 다운로드 API 수정** (filtered 버전 제거):
```python
# app/api/jobs.py - get_ply_file() 함수
@router.get("/pub/{pub_key}/cloud.ply")
async def get_ply_file(pub_key: str, db: Session = Depends(get_db)):
    job = crud.get_job_by_pub_key(db, pub_key)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    ply_dir = Path(f"data/jobs/{job.job_id}/output/point_cloud/iteration_{job.iterations}")
    
    # filtered 버전 제거, 원본만 제공
    ply_file = ply_dir / "point_cloud.ply"
    ply_gz = ply_dir / "point_cloud.ply.gz"
    
    # GZIP 버전 우선 제공
    if ply_gz.exists():
        return FileResponse(
            ply_gz,
            media_type="application/octet-stream",
            headers={
                "Content-Encoding": "gzip",
                "Content-Disposition": "attachment; filename=point_cloud.ply",
                "Cache-Control": "public, max-age=31536000"
            }
        )
    
    return FileResponse(ply_file, media_type="application/octet-stream")
```

#### 📊 효과
- **절약 시간**: 작업당 10-20초
- **코드 간소화**: K-NN, DBSCAN 알고리즘 제거
- **파일 간소화**: `point_cloud_filtered.ply` 생성 안 함

**커밋**:
```bash
git add app/core/gaussian_splatting.py app/api/jobs.py
git commit -m "refactor: remove outlier filtering (10-20s saved, minimal quality impact)"
```

---

### 5. COLMAP Validation 간소화 ⭐⭐

**파일**: `app/utils/colmap_validator.py`, `app/api/jobs.py`

#### 🤔 왜 간소화하나요?

**현재 상황**:
```python
# app/utils/colmap_validator.py - validate_colmap_reconstruction()
def validate_colmap_reconstruction(sparse_dir: str):
    # 5개 조건 체크 (모두 통과해야 함)
    
    # 1. 등록된 이미지 수 >= 3장 (Error)
    # 2. 이미지 등록률 >= 60% (Error), >= 80% (Warning)
    # 3. 3D 포인트 수 >= 300개 (Error), >= 800개 (Warning)
    # 4. 평균 트랙 길이 >= 2.0 (Error), >= 3.0 (Warning)
    # 5. 포인트/이미지 비율 >= 80 (Error), >= 100 (Warning)
    
    # 하나라도 Error 조건 실패 시 작업 전체 실패
```

**문제점**:
1. **너무 엄격함**: 5개 조건 중 하나만 실패해도 전체 실패
2. **불필요한 실패**: 품질 낮은 재구성도 3D 뷰어로는 괜찮을 수 있음
3. **해커톤 리스크**: 테스트 이미지로 실패 가능성 높음
4. **사용자 경험 저하**: "재구성 실패" 메시지만 보고 원인 모름

**해결 방법**:
- **최소 검증만 수행** (2개 조건만)
- Warning은 로그만 출력하고 통과
- 사용자가 3D 뷰어로 직접 판단

#### 🔧 작업

```python
# app/utils/colmap_validator.py - simple_validation()
"""
간소화된 COLMAP 검증
- 최소 2개 조건만 체크 (필수 파일, 최소 이미지 수)
- Warning은 로그만 출력 (작업은 통과)
"""
from pathlib import Path
from app.utils.logger import logger

def simple_validation(sparse_dir: str) -> bool:
    """
    간소화된 COLMAP 검증 (MVP용)
    
    Args:
        sparse_dir: COLMAP sparse 재구성 결과 디렉토리 (예: work/sparse/0)
    
    Returns:
        bool: 검증 통과 여부
    
    Raises:
        ValidationError: 필수 조건 실패 시
    """
    sparse_path = Path(sparse_dir)
    
    # === 필수 조건 1: 파일 존재 확인 ===
    cameras_file = sparse_path / "cameras.txt"
    images_file = sparse_path / "images.txt"
    points_file = sparse_path / "points3D.txt"
    
    missing_files = []
    if not cameras_file.exists():
        missing_files.append("cameras.txt")
    if not images_file.exists():
        missing_files.append("images.txt")
    if not points_file.exists():
        missing_files.append("points3D.txt")
    
    if missing_files:
        raise ValidationError(
            f"COLMAP reconstruction failed: missing files {missing_files}"
        )
    
    # === 필수 조건 2: 최소 이미지 수 (3장) ===
    registered_count = count_registered_images(images_file)
    
    if registered_count < 3:
        raise ValidationError(
            f"Too few registered images: {registered_count}/3 minimum. "
            f"COLMAP failed to register enough images."
        )
    
    # === Warning 조건 (통과시킴) ===
    if registered_count < 5:
        logger.warning(
            f"⚠️  Low image count: {registered_count} registered "
            f"(recommended: 5+). Quality may be affected."
        )
    
    # === 추가 정보 로깅 (참고용) ===
    try:
        point_count = count_3d_points(points_file)
        logger.info(f"COLMAP statistics:")
        logger.info(f"  - Registered images: {registered_count}")
        logger.info(f"  - 3D points: {point_count}")
    except Exception as e:
        logger.warning(f"Failed to read COLMAP statistics: {e}")
    
    logger.info("✓ COLMAP validation passed (minimal checks)")
    return True


def count_registered_images(images_file: Path) -> int:
    """images.txt에서 등록된 이미지 수 계산"""
    count = 0
    with open(images_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            # 홀수 줄만 카운트 (이미지 정보)
            count += 1
    return count // 2  # 2줄이 1개 이미지


def count_3d_points(points_file: Path) -> int:
    """points3D.txt에서 3D 포인트 수 계산"""
    count = 0
    with open(points_file, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            count += 1
    return count


class ValidationError(Exception):
    """COLMAP 검증 실패 시 발생하는 예외"""
    pass
```

**API에서 호출 방식 변경**:
```python
# app/api/jobs.py - process_job() 함수
try:
    update_job_status(db, job_id, "PROCESSING", "COLMAP_VALIDATE", 60)
    
    # 기존 복잡한 검증 대신 간소화된 검증 사용
    from app.utils.colmap_validator import simple_validation
    simple_validation(sparse_dir)
    
except ValidationError as e:
    logger.error(f"COLMAP validation failed: {e}")
    handle_job_error(db, job_id, "COLMAP_VALIDATE", str(e), traceback.format_exc())
    return
```

#### 📊 효과
- **실패 감소**: 엄격한 조건 제거
- **유연성 향상**: 품질 낮은 재구성도 허용 (사용자 판단)
- **로그 개선**: Warning으로 추가 정보 제공

**커밋**:
```bash
git add app/utils/colmap_validator.py app/api/jobs.py
git commit -m "refactor: simplify COLMAP validation (2 checks only, more flexible)"
```

---

### 6. 이미지 리사이징 최적화 ⭐⭐⭐

**파일**: `app/config.py`

#### 🤔 왜 변경하나요?

**현재 상황**:
```python
# app/config.py
class Settings(BaseSettings):
    MAX_IMAGE_SIZE: int = 1600  # 기존 설정
```

**문제점 / 개선 여지**:
1. **처리 시간**: 1600px는 충분히 크지만 더 빠를 수 있음
2. **GPU 메모리**: 해상도 ↓ → GPU 메모리 절약
3. **COLMAP 안정성**: 너무 큰 이미지는 매칭 실패율 증가

**연구 결과** (이전 분석):
- **1280px**: COLMAP 최적화, 20-30% 빠름, 품질 손실 거의 없음
- **1600px** (현재): 균형잡힌 선택, 안정적
- **2048px**: 고품질이지만 느림, 매칭 불안정
- **4K+**: 매우 느림, 실패율 높음

**결론**:
- **해커톤 MVP**: **1280px로 변경** (속도 우선)
- **프로덕션**: 1600px 유지 가능 (품질 우선)

#### 🔧 작업

```python
# app/config.py
class Settings(BaseSettings):
    # ... 기존 설정 ...
    
    # Image processing
    MAX_IMAGE_SIZE: int = 1280  # 변경: 1600 → 1280
    
    # 이유:
    # 1. COLMAP 매칭 최적화 (연구 기반)
    # 2. GPU 메모리 절약 (~30% 감소)
    # 3. 처리 시간 20-30% 단축
    # 4. 품질 손실 거의 없음 (실증 연구)
    # 
    # 참고: 프로덕션에서는 1600px로 복원 가능
```

#### 📊 효과
- **처리 시간**: 20-30% 단축 (COLMAP + GS 합산)
  - COLMAP: 3-7분 → 2-5분
  - GS Training: 4-6분 → 3-5분
- **GPU 메모리**: ~30% 절약
- **품질**: 거의 동일 (사용자 체감 어려움)

**커밋**:
```bash
git add app/config.py
git commit -m "perf: optimize image resize to 1280px (20-30% faster, minimal quality loss)"
```

---

### 7. Queue Status API 유지 ✅ (변경 없음)

**파일**: `app/api/jobs.py`

#### 🤔 왜 유지하나요?

**기능**:
```python
@router.get("/queue", response_model=QueueStatusResponse)
async def get_queue_status(db: Session = Depends(get_db)):
    """작업 대기열 상태 조회"""
    return {
        "max_concurrent": settings.MAX_CONCURRENT_JOBS,  # 1
        "running_count": 1,
        "pending_count": 2,
        "running_jobs": [{"job_id": "abc", "started_at": "..."}],
        "pending_jobs": [
            {"job_id": "def", "position": 1},  # 대기 1번째
            {"job_id": "ghi", "position": 2}   # 대기 2번째
        ]
    }
```

**필요한 이유**:
1. **Semaphore 기반 동시성 제어**:
   - `MAX_CONCURRENT_JOBS = 1` (GPU 메모리 제약)
   - 동시에 1개 작업만 처리
   - 나머지는 Queue에서 대기

2. **실제 사용 시나리오**:
   ```
   [시간축] -------------------------------->
   
   User A 작업 시작 (10분 소요)
   ├─ COLMAP (3분)
   ├─ GS Training (5분)
   └─ 완료
   
            User B 작업 Queue 대기 (position: 1)
            └─ "1번째 대기 중..."
   
                     User C 작업 Queue 대기 (position: 2)
                     └─ "2번째 대기 중..."
   
   User A 완료 → User B 자동 시작 (Semaphore 해제)
   ```

3. **Frontend 활용**:
   ```typescript
   // 내 판매 내역 페이지
   const ProductStatus = ({ product }) => {
     const [queueInfo, setQueueInfo] = useState(null);
     
     useEffect(() => {
       if (product.status === 'QUEUED') {
         const interval = setInterval(async () => {
           const queue = await api.getQueueStatus();
           const position = queue.pending_jobs.findIndex(
             j => j.job_id === product.job_id
           ) + 1;
           setQueueInfo({ position, total: queue.pending_count });
         }, 5000);  // 5초마다 확인
         
         return () => clearInterval(interval);
       }
     }, [product.status]);
     
     if (product.status === 'QUEUED') {
       return (
         <div>
           ⏳ 대기 중 ({queueInfo?.position}/{queueInfo?.total}번째)
           <p>현재 {queue.running_count}개 작업 처리 중</p>
         </div>
       );
     }
   };
   ```

4. **사용자 경험 향상**:
   - "처리 중" (막연함) → "2번째 대기 중" (구체적)
   - 대기 시간 예상 가능
   - 불안감 감소

#### 📊 결론
- **변경 없음**: 기존 코드 그대로 유지
- **이유**: MVP에서도 필수 기능 (Queue 관리)
- **Frontend 구현 권장**: 대기 순서 표시

---

### 8. ErrorLog 테이블 유지 ✅ (변경 없음)

**파일**: `app/db/models.py`

#### 🤔 왜 유지하나요?

**기능**:
```python
class ErrorLog(Base):
    __tablename__ = "error_logs"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    job_id = Column(String, ForeignKey("jobs.job_id"))
    timestamp = Column(DateTime, default=datetime.utcnow)
    stage = Column(String, nullable=False)  # COLMAP_FEAT, GS_TRAIN 등
    error_type = Column(String, nullable=False)
    error_message = Column(Text, nullable=False)
    traceback = Column(Text, nullable=True)
```

**필요한 이유**:
1. **디버깅**: 실패 원인 추적 용이
2. **통계**: 어느 단계에서 자주 실패하는지 분석
3. **이미 구현됨**: 삭제 시 오히려 복잡해짐
4. **Job 테이블 간소화**: error_message 컬럼만으로는 부족

#### 📊 결론
- **변경 없음**: 기존 코드 그대로 유지
- **이유**: 유용하고, 삭제 시 이득 없음

---

## 📊 최종 변경 요약

| 작업 | 상태 | 절약 시간 | 이유 |
|------|------|----------|------|
| **1. Preflight Check 제거** | ✅ 제거 | 1-2초 | 서버 시작 시 1회만 검사하면 충분 |
| **2. Evaluation 제거** | ✅ 제거 | 30-60초 | DB에 메트릭 컬럼 없음, 사용자 불필요 |
| **3. Train/Test Split 제거** | ✅ 제거 | 5-10초 | Evaluation 안 하므로 불필요 |
| **4. Outlier Filtering 제거** | ✅ 제거 | 10-20초 | GS 자체가 노이즈에 강함, 효과 미미 |
| **5. COLMAP Validation 간소화** | 🔵 간소화 | - | 2개 조건만 체크, 유연성 향상 |
| **6. 이미지 리사이징 (1280px)** | 🔵 변경 | 2-3분 | 연구 기반, 20-30% 빠름, 품질 동일 |
| **7. Queue Status API** | ⛔ 유지 | - | **필수 기능** (Queue 관리) |
| **8. ErrorLog 테이블** | ⛔ 유지 | - | 디버깅 용이, 삭제 이득 없음 |

### 총 효과
- **절약 시간**: **3-5분** (작업당 30-40% 단축)
- **기존**: 10-15분
- **최적화 후**: **7-10분**

---

## 🧪 테스트 절차

각 작업 완료 후 **반드시 테스트**:

```bash
# 1. 서버 시작 (Preflight check 확인)
python main.py
# 출력:
# ==================================================
# Running preflight check on server startup...
# ==================================================
# ✅ Preflight check passed! Server is ready.
# ==================================================

# 2. 테스트 작업 생성
curl -X POST http://localhost:8001/recon/jobs \
  -F "files=@test1.jpg" \
  -F "files=@test2.jpg" \
  -F "files=@test3.jpg"

# 응답 예시:
# {
#   "job_id": "rU91efWW",
#   "pub_key": "wsqZK46RiH",
#   "original_resolution": false
# }

# 3. 작업 상태 확인 (진행 중)
curl http://localhost:8001/recon/jobs/rU91efWW/status

# 4. Queue 상태 확인
curl http://localhost:8001/recon/queue

# 5. 작업 완료 대기 (7-10분)

# 6. 3D 뷰어 확인
http://localhost:8001/v/wsqZK46RiH

# 7. PLY 다운로드
curl -O http://localhost:8001/recon/pub/wsqZK46RiH/cloud.ply
```

### 검증 항목 체크리스트:

- [ ] **서버 시작**: Preflight check 1회만 실행
- [ ] **작업 생성**: 3장 이상 이미지 업로드 성공
- [ ] **COLMAP**: 3장 이상 등록 확인 (로그)
- [ ] **GS Training**: 7000 iterations 완료
- [ ] **PLY 생성**: `point_cloud.ply` 존재 (filtered 없음)
- [ ] **GZIP**: `point_cloud.ply.gz` 존재
- [ ] **3D 뷰어**: 정상 로드 및 렌더링
- [ ] **DB**: psnr, ssim, lpips 컬럼 없음 (마이그레이션 확인)
- [ ] **Queue**: 여러 작업 동시 등록 시 대기열 동작
- [ ] **처리 시간**: 7-10분 (기존 대비 30-40% 단축)

---

## 📝 문서 업데이트

**작업 완료 후 README.md 업데이트:**

```markdown
# InstaRecon3D (MVP Edition)

## 🎯 MVP 최적화 (Hackathon Edition)

해커톤 MVP에 맞춰 불필요한 기능을 제거하고 성능을 최적화했습니다.

### ❌ 제거된 기능
- **Per-job Preflight Check** → 서버 시작 시 1회만 실행
- **Model Evaluation** (PSNR, SSIM, LPIPS) → 사용자 불필요
- **Train/Test Split** → Evaluation 제거로 불필요
- **Outlier Filtering** (K-NN + DBSCAN) → 효과 미미, GS 자체가 강함

### 🔵 간소화된 기능
- **COLMAP Validation** → 최소 2개 조건만 체크 (유연성 향상)
- **Image Resize** → 1280px (기존 1600px, 연구 기반 최적화)

### ⛔ 유지된 기능
- **Queue Status API** → Semaphore 기반 동시성 제어 (필수)
- **ErrorLog Table** → 디버깅 및 통계 수집

### ⚡ 성능 개선
- **처리 시간**: 10-15분 → **7-10분** (30-40% 단축)
- **GPU 메모리**: ~30% 절약
- **디스크 공간**: Train/Test, Filtered PLY 불필요

### 📊 변경 사항 상세

| 항목 | 기존 | 변경 후 | 효과 |
|------|------|---------|------|
| Preflight | 매 작업마다 | 서버 시작 시 1회 | -1~2초 |
| Evaluation | 필수 | 제거 | -30~60초 |
| Train/Test Split | 80/20 분할 | 제거 | -5~10초 |
| Outlier Filter | K-NN + DBSCAN | 제거 | -10~20초 |
| Image Resize | 1600px | 1280px | -2~3분 |
| COLMAP Validation | 5개 조건 | 2개 조건 | 유연성↑ |

### 🔮 향후 작업 (Backend 통합 후)
- Callback URL 지원
- S3 Image Download
- Job ID 외부 입력
- DB 동기화 (Backend Polling)

---

## 📖 사용 방법

### 서버 시작
```bash
python main.py
# Preflight check가 1회 실행되고 서버 시작
```

### 작업 생성
```bash
curl -X POST http://localhost:8001/recon/jobs \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

### Queue 확인
```bash
curl http://localhost:8001/recon/queue
```

### 3D 뷰어
```
http://localhost:8001/v/{pub_key}
```

---

## 🧪 테스트

최소 3장 이상의 이미지로 전체 파이프라인 테스트:
```bash
# 테스트 이미지 준비 (3-20장)
# 다양한 각도에서 촬영한 이미지 권장

python main.py
# 작업 생성 및 완료 대기 (7-10분)
```
```

**커밋**:
```bash
git add README.md
git commit -m "docs: update README with MVP changes and optimization details"
```

---

## ✅ 최종 체크리스트

작업 완료 전 확인:

- [ ] **모든 코드 변경 완료**
  - [ ] Preflight → main.py로 이동
  - [ ] Evaluation 제거 + DB 마이그레이션
  - [ ] Train/Test Split 제거
  - [ ] Outlier Filtering 제거
  - [ ] COLMAP Validation 간소화
  - [ ] Image Resize 1280px

- [ ] **테스트 통과**
  - [ ] 서버 시작 (Preflight 1회)
  - [ ] 작업 생성 (3장)
  - [ ] COLMAP 성공
  - [ ] GS Training 성공
  - [ ] PLY 생성 (filtered 없음)
  - [ ] 3D 뷰어 로드
  - [ ] Queue 동작 확인

- [ ] **DB 마이그레이션**
  - [ ] 메트릭 컬럼 제거 완료
  - [ ] 기존 데이터 보존 확인

- [ ] **문서 업데이트**
  - [ ] README.md 업데이트
  - [ ] 변경 사항 상세 기록

- [ ] **Git 정리**
  - [ ] 모든 변경사항 커밋
  - [ ] 브랜치 푸시: `git push origin feature/mvp-refactor`

---

## 🔮 Phase 2: Backend 통합 (향후)

**Backend API 서버 구현 후 추가할 기능:**

### 1. Callback URL 지원
```python
# POST /recon/jobs에 callback_url 파라미터 추가
@router.post("/jobs")
async def create_job(..., callback_url: Optional[str] = None):
    # ...
    
    # 작업 완료 시 Backend에 POST
    if callback_url:
        requests.post(callback_url, json={
            "job_id": job_id,
            "status": "DONE",
            "ply_url": f"{BASE_URL}/recon/pub/{pub_key}/cloud.ply",
            "progress": 100
        })
```

### 2. S3 Image Download
```python
# Backend가 S3 URL 전달 → 3DGS 서버가 다운로드
@router.post("/jobs")
async def create_job(image_urls: List[str], ...):
    for url in image_urls:
        response = await httpx.get(url)
        save_image(response.content, ...)
```

### 3. Job ID 외부 입력
```python
# Backend가 Job ID 생성 → 3DGS 서버는 받아서 사용
@router.post("/jobs")
async def create_job(job_id: str, ...):
    # Backend가 생성한 ID 사용
    # 충돌 방지
```

### 4. DB 동기화
```python
# Backend가 3DGS 서버 API로 Job 상태 조회
# Polling 방식 (5초마다)
const jobStatus = await fetch(`${API_URL}/recon/jobs/${jobId}/status`);
```

---

## 📞 문의 및 문제 해결

### 작업 중 문제 발생 시:

1. **Git 상태 확인**:
   ```bash
   git status
   git log --oneline -10
   ```

2. **로그 확인**:
   ```bash
   # 서버 로그
   tail -f logs/server.log
   
   # 작업 로그
   tail -f data/jobs/{job_id}/logs/process.log
   ```

3. **DB 확인**:
   ```bash
   sqlite3 data/jobs.db
   .schema jobs
   SELECT * FROM jobs ORDER BY created_at DESC LIMIT 5;
   ```

4. **디스크 공간 확인**:
   ```bash
   df -h
   du -sh data/jobs/*
   ```

### 일반적인 문제:

| 문제 | 원인 | 해결 |
|------|------|------|
| Preflight 실패 | CUDA 미설치 | nvidia-smi 확인 |
| COLMAP 실패 | 이미지 < 3장 | 최소 3장 업로드 |
| GS OOM | GPU 메모리 부족 | MAX_CONCURRENT_JOBS=1 확인 |
| PLY 없음 | 학습 실패 | 로그 확인 |

---

## 🎉 완료!

**중요**: 각 작업 후 **반드시 커밋**하고, **테스트** 후 다음 작업 진행!

**최종 목표**:
- ✅ 처리 시간 30-40% 단축 (10-15분 → 7-10분)
- ✅ 코드 간소화 및 안정성 향상
- ✅ 해커톤 MVP에 최적화

**Git 브랜치**:
```bash
git push origin feature/mvp-refactor
# → 이후 main 브랜치에 머지
```

**다음 단계**:
Backend API 서버 구현 후 Phase 2 작업 진행!
