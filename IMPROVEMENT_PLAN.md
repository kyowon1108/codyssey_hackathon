# 시스템 개선 계획서

## ✅ 완료된 작업

### 1. 데이터베이스 인프라 구축
- **파일**: `database.py`
- **내용**:
  - SQLAlchemy 기반 데이터베이스 모델 생성
  - Job 모델: 작업 정보, 상태, 통계, 에러 추적
  - ErrorLog 모델: 상세한 에러 로깅
  - CRUD 함수: create_job, get_job_by_id, update_job_status, etc.

### 2. Requirements 업데이트
- SQLAlchemy 추가

## 🚧 진행 중인 작업

### 3. main.py 데이터베이스 마이그레이션
현재 `main.py`는 메모리 기반 `jobs`, `pub_to_job` 딕셔너리를 사용 중입니다.
이를 데이터베이스 기반으로 전환해야 합니다.

**필요한 수정사항**:

#### A. 전역 변수 제거
```python
# 제거할 코드:
jobs = {}  # job_id -> 정보(dict: status, pub_key, etc.)
pub_to_job = {}  # pub_key -> job_id 매핑

# 추가할 코드:
# Database initialization
init_db()  # 데이터베이스 테이블 생성
```

#### B. recover_existing_jobs() 함수 수정
```python
# 기존: 메모리에 복구
# 수정: 데이터베이스에서 자동 복구 (DB에 이미 저장되어 있음)
def recover_existing_jobs():
    """
    서버 재시작 시 DB에서 RUNNING 상태인 작업들을 PENDING으로 변경
    """
    db = next(get_db())
    running_jobs = get_running_jobs(db)
    for job in running_jobs:
        logger.warning(f"Found interrupted job: {job.job_id}, resetting to PENDING")
        update_job_status(db, job.job_id, "PENDING")
    db.close()
```

#### C. create_reconstruction_job() API 수정
```python
@app.post("/recon/jobs")
async def create_reconstruction_job(
    files: list[UploadFile] = File(...),
    original_resolution: bool = False,
    db: Session = Depends(get_db)  # 데이터베이스 세션 주입
):
    # 1. 이미지 검증 추가
    for file in files:
        validate_image_file(file)  # 새로운 검증 함수

    # 2. Job ID 생성
    job_id = uuid.uuid4().hex[:8]
    pub_key = ''.join(random.choice("0123456789") for _ in range(10))

    # 3. 데이터베이스에 저장
    try:
        job = create_job(
            db,
            job_id=job_id,
            pub_key=pub_key,
            image_count=len(files),
            original_resolution=original_resolution,
            iterations=10000
        )
    except Exception as e:
        logger.error(f"Failed to create job in database: {e}")
        raise HTTPException(500, "Failed to create job")

    # 4. 이미지 저장 및 작업 시작
    # ... (기존 코드)

    # 5. 백그라운드 작업 시작
    asyncio.create_task(process_job(job_id))

    return {
        "job_id": job_id,
        "pub_key": pub_key,
        "original_resolution": original_resolution
    }
```

#### D. process_job() 함수 에러 핸들링 강화
```python
async def process_job(job_id: str):
    """
    주어진 job_id에 대해 COLMAP 및 Gaussian Splatting 파이프라인을 실행합니다.
    """
    db = next(get_db())
    job = get_job_by_id(db, job_id)

    if not job:
        logger.error(f"Job {job_id} not found in database")
        return

    async with sem:
        job_dir = Path(BASE_DIR / job_id)
        log_path = job_dir / "run.log"

        try:
            # 상태 업데이트: RUNNING
            update_job_status(db, job_id, "RUNNING")

            with open(log_path, 'a', encoding='utf-8', buffering=1) as log_file:
                log_file.write(f"=== Job {job_id} started ===\n")

                # ========== COLMAP Feature Extraction ==========
                try:
                    log_file.write(">> [1/6] Feature extraction...\n")
                    await run_command([...], log_file)
                except Exception as e:
                    error_msg = f"Feature extraction failed: {str(e)}"
                    log_error(db, job_id, "colmap_feature", type(e).__name__, error_msg, traceback.format_exc())
                    raise

                # ========== COLMAP Feature Matching ==========
                try:
                    log_file.write(">> [2/6] Feature matching...\n")
                    await run_command([...], log_file)
                except Exception as e:
                    error_msg = f"Feature matching failed: {str(e)}"
                    log_error(db, job_id, "colmap_matching", type(e).__name__, error_msg, traceback.format_exc())
                    raise

                # ... 각 단계마다 try-except 추가

                # ========== Gaussian Splatting Training ==========
                try:
                    log_file.write(">> [5/6] Gaussian Splatting training...\n")
                    await run_command(gs_train_cmd, log_file, env=env, monitor_gpu=True)

                    # 통계 업데이트
                    ply_file = iter_dir / "point_cloud_filtered.ply"
                    if ply_file.exists():
                        gaussian_count = count_gaussians(ply_file)
                        file_size_mb = ply_file.stat().st_size / (1024 * 1024)

                        update_job_results(
                            db,
                            job_id,
                            gaussian_count=gaussian_count,
                            file_size_mb=file_size_mb
                        )

                except Exception as e:
                    error_msg = f"Gaussian Splatting failed: {str(e)}"
                    log_error(db, job_id, "gaussian_splatting", type(e).__name__, error_msg, traceback.format_exc())
                    raise

                # 완료 상태 업데이트
                update_job_status(db, job_id, "DONE")
                log_file.write(f"=== Job {job_id} completed successfully ===\n")

        except Exception as e:
            # 전체 작업 실패 처리
            error_msg = f"Job failed: {str(e)}"
            logger.error(error_msg)
            log_error(db, job_id, "general", type(e).__name__, error_msg, traceback.format_exc())
            update_job_status(db, job_id, "FAILED", error_message=str(e))

        finally:
            db.close()
```

#### E. get_job_status() API 수정
```python
@app.get("/recon/jobs/{job_id}/status")
async def get_job_status(job_id: str, db: Session = Depends(get_db)):
    job = get_job_by_id(db, job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    # 로그 파일 읽기
    log_path = Path(BASE_DIR / job_id / "run.log")
    log_tail = []
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                log_tail = lines[-10:] if lines else []
        except Exception as e:
            log_tail = [f"(log read error: {e})"]

    response = {
        "job_id": job.job_id,
        "status": job.status,
        "log_tail": log_tail,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "processing_time_seconds": job.processing_time_seconds,
        "image_count": job.image_count,
        "gaussian_count": job.gaussian_count,
        "file_size_mb": job.file_size_mb
    }

    if job.status == "DONE":
        response["viewer_url"] = f"/v/{job.pub_key}"
    elif job.status == "FAILED":
        response["error"] = job.error_message
        response["error_stage"] = job.error_stage

    return JSONResponse(content=response)
```

#### F. view_result_page() API 수정
```python
@app.get("/v/{pub_key}", response_class=HTMLResponse)
async def view_result_page(pub_key: str, request: Request, db: Session = Depends(get_db)):
    job = get_job_by_pub_key(db, pub_key)

    if not job:
        raise HTTPException(status_code=404, detail="Invalid viewer key")

    if job.status != "DONE":
        return HTMLResponse(content="<html><body><h3>Result is not ready yet. Please check again later.</h3></body></html>")

    # Load viewer
    viewer_path = Path(__file__).parent / "viewer_template.html"
    with open(viewer_path, 'r') as f:
        html_content = f.read()

    splat_url = f"/recon/pub/{pub_key}/scene.splat"
    html_content = html_content.replace('SPLAT_URL_PLACEHOLDER', splat_url)
    return HTMLResponse(content=html_content, status_code=200)
```

#### G. 이미지 검증 함수 추가
```python
def validate_image_file(file: UploadFile):
    """
    업로드된 파일이 유효한 이미지인지 검증합니다.
    """
    from PIL import Image
    import io

    # 파일 확장자 체크
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    file_ext = Path(file.filename).suffix

    if file_ext not in allowed_extensions:
        raise HTTPException(400, f"Unsupported file type: {file_ext}. Allowed: {allowed_extensions}")

    # 파일 내용 검증
    try:
        content = file.file.read()
        img = Image.open(io.BytesIO(content))
        img.verify()  # 이미지 파일 무결성 검증

        # 파일 포인터 리셋
        file.file.seek(0)

        # 최소 크기 체크
        if img.width < 100 or img.height < 100:
            raise HTTPException(400, f"Image too small: {img.width}x{img.height}. Minimum: 100x100")

        logger.info(f"Validated image: {file.filename} ({img.width}x{img.height})")

    except Exception as e:
        raise HTTPException(400, f"Invalid image file: {file.filename}. Error: {str(e)}")
```

#### H. Gaussian 카운팅 함수 추가
```python
def count_gaussians(ply_file: Path) -> int:
    """
    PLY 파일에서 Gaussian 개수를 카운트합니다.
    """
    try:
        from plyfile import PlyData
        plydata = PlyData.read(str(ply_file))
        return len(plydata['vertex'])
    except Exception as e:
        logger.error(f"Failed to count gaussians: {e}")
        return 0
```

## 📋 남은 작업

### 우선순위 1: 핵심 마이그레이션
- [ ] main.py에 데이터베이스 연동 완료
- [ ] 모든 API 엔드포인트 DB 기반으로 전환
- [ ] 에러 핸들링 강화 완료

### 우선순위 2: 테스트 및 검증
- [ ] 새 이미지 업로드 테스트
- [ ] 서버 재시작 후 작업 복구 테스트
- [ ] 에러 발생 시 로그 확인

### 우선순위 3: 추가 기능
- [ ] 작업 취소 API
- [ ] 작업 삭제 API
- [ ] 관리자 대시보드 (모든 작업 조회)

## 🔧 적용 방법

### 방법 1: 점진적 마이그레이션 (권장)
1. 현재 main.py를 main_old.py로 백업
2. 새로운 main.py 작성 (DB 기반)
3. 테스트
4. 문제 없으면 main_old.py 삭제

### 방법 2: 하이브리드 방식
1. DB를 추가하되, 메모리 캐시도 유지
2. 성능과 안정성 확보
3. 나중에 메모리 캐시 제거

## 📊 예상 효과

### Before (메모리 기반)
- ❌ 서버 재시작 시 작업 정보 손실
- ❌ 에러 추적 어려움
- ❌ 통계 데이터 없음
- ❌ 작업 히스토리 없음

### After (DB 기반)
- ✅ 영구 저장, 서버 재시작 안전
- ✅ 상세한 에러 로깅 및 추적
- ✅ 통계 및 분석 가능
- ✅ 작업 히스토리 및 검색 가능
- ✅ 상용 서비스 준비 완료

## 🚀 다음 단계

1. **SQLAlchemy 설치**:
   ```bash
   pip install sqlalchemy
   ```

2. **데이터베이스 초기화**:
   ```python
   from database import init_db
   init_db()
   ```

3. **main.py 수정 적용**

4. **테스트**

5. **Git commit**

---

**작성일**: 2025-10-12
**작성자**: Claude Code
