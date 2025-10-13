# InstaRecon3D - Gaussian Splatting 3D Reconstruction API

FastAPI 기반 3D 재구성 서비스로, 여러 이미지를 업로드하면 COLMAP과 Gaussian Splatting을 사용하여 3D 모델을 자동 생성하고 웹 뷰어로 확인할 수 있습니다.

## 목차

- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [빠른 시작](#빠른-시작)
- [개발 환경 설정](#개발-환경-설정)
- [프로젝트 구조](#프로젝트-구조)
- [API 사용법](#api-사용법)
- [개발 가이드](#개발-가이드)
- [브랜치 전략](#브랜치-전략)
- [테스트](#테스트)
- [문제 해결](#문제-해결)

## 주요 기능

- **이미지 업로드 & 3D 재구성**: 3~20장의 이미지로 3D 모델 자동 생성
- **단계별 진행률 추적**: COLMAP → Gaussian Splatting → 후처리 전체 과정 실시간 모니터링
- **작업 대기열 시스템**: asyncio.Semaphore 기반 순차 처리 (GPU 메모리 및 포트 충돌 방지)
- **업로드 검증**: 파일 크기(개별 30MB, 전체 500MB), MIME 타입, 이미지 개수 자동 검증
- **아웃라이어 필터링**: K-NN + DBSCAN 기반 노이즈 제거
- **웹 3D 뷰어**: Splat 형식으로 최적화된 실시간 렌더링
- **Preflight 체크**: Python, CUDA, COLMAP, 파일시스템 사전 검증
- **Health Check**: `/health`, `/healthz` 엔드포인트 제공

## 시스템 요구사항

- **OS**: Ubuntu 22.04 (권장)
- **Python**: 3.9+
- **GPU**: CUDA 지원 GPU (12.8+ 권장, RTX 4060 Ti 16GB 검증 완료)
- **디스크**: 작업당 약 1-5GB 여유 공간
- **RAM**: 최소 16GB 권장

## 빠른 시작

```bash
# 1. 레포지토리 클론
git clone <repository-url>
cd codyssey_hackathon

# 2. Conda 환경 생성 및 활성화
conda create -n codyssey python=3.9
conda activate codyssey

# 3. 의존성 설치
pip install -r requirements.txt

# 4. COLMAP 설치 (Ubuntu)
sudo apt-get update
sudo apt-get install colmap

# 5. Gaussian Splatting 설정
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
cd ..

# 6. 서버 실행
python main.py
```

서버가 `http://0.0.0.0:8000`에서 실행됩니다.

## 개발 환경 설정

### 1. 필수 의존성 설치

```bash
# PyTorch (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 또는 CUDA 12.1+
pip install torch torchvision

# scikit-learn (아웃라이어 필터링)
pip install scikit-learn

# FastAPI & 기타
pip install fastapi uvicorn sqlalchemy python-multipart pillow plyfile aiofiles
```

### 2. 환경 변수 설정 (선택)

`.env` 파일을 생성하여 설정을 오버라이드할 수 있습니다:

```bash
# .env
HOST=0.0.0.0
PORT=8000
BASE_URL=http://localhost:8000
DEBUG=false

MAX_CONCURRENT_JOBS=1
TRAINING_ITERATIONS=10000
```

### 3. 데이터베이스 초기화

서버를 처음 실행하면 `gaussian_splatting.db`가 자동으로 생성됩니다. 수동으로 초기화하려면:

```bash
python -c "from app.db.database import init_db; init_db()"
```

### 4. Preflight Check 실행

시스템이 준비되었는지 확인:

```bash
python -c "from app.utils.preflight import run_preflight_check; result = run_preflight_check(); print(result.get_summary())"
```

## 프로젝트 구조

```
codyssey_hackathon/
├── app/                          # FastAPI 애플리케이션
│   ├── api/                      # API 엔드포인트
│   │   ├── jobs.py              # 작업 관리 (생성, 상태 조회, 대기열)
│   │   └── viewer.py            # 3D 뷰어 렌더링
│   ├── core/                     # 핵심 처리 로직
│   │   ├── colmap.py            # COLMAP 파이프라인
│   │   └── gaussian_splatting.py # Gaussian Splatting 학습
│   ├── db/                       # 데이터베이스
│   │   ├── models.py            # ORM 모델 (Job, ErrorLog)
│   │   ├── crud.py              # CRUD 함수
│   │   └── database.py          # SQLAlchemy 설정
│   ├── schemas/                  # Pydantic 스키마
│   │   └── job.py               # API 요청/응답 검증
│   ├── utils/                    # 유틸리티 함수
│   │   ├── preflight.py         # 환경 사전 점검 (NEW)
│   │   ├── converter.py         # PLY → Splat 변환
│   │   ├── outlier_filter.py   # 아웃라이어 필터링
│   │   ├── image.py             # 이미지 검증/리사이징
│   │   ├── logger.py            # 로깅 설정
│   │   └── system.py            # GPU 모니터링
│   ├── config.py                 # 전역 설정
│   └── main.py                   # FastAPI 진입점
│
├── main.py                       # 서버 실행 파일
├── requirements.txt             # Python 의존성
├── IMPLEMENT.md                 # 구현 명세서 (Refactoring)
├── templates/                    # HTML 템플릿
│   └── viewer.html              # 3D 뷰어 페이지
├── gaussian-splatting/          # Gaussian Splatting 레포지토리
├── data/jobs/                   # 작업 데이터 저장소
└── gaussian_splatting.db        # SQLite 데이터베이스
```

### 데이터 디렉토리 구조

각 작업은 `data/jobs/{job_id}/` 디렉토리에 저장됩니다:

```
data/jobs/{job_id}/
├── upload/images/              # 업로드된 원본 이미지
├── colmap/                     # COLMAP 처리 결과
│   ├── database.db
│   └── sparse/0/
├── work/                       # 중간 처리 데이터
│   ├── images/                 # 언디스토션된 이미지
│   └── sparse/0/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── output/                     # 최종 결과물
│   └── point_cloud/
│       └── iteration_{N}/
│           ├── point_cloud.ply
│           ├── point_cloud_filtered.ply
│           └── scene.splat
└── logs/
    └── process.log            # 작업 처리 로그
```

## API 사용법

### 엔드포인트 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | API 정보 |
| GET | `/health` | Health check (JSON) |
| GET | `/healthz` | Health check (Plain text "ok") |
| POST | `/recon/jobs` | 새 작업 생성 |
| GET | `/recon/jobs/{job_id}/status` | 작업 상태 조회 |
| GET | `/recon/queue` | 대기열 상태 |
| GET | `/recon/pub/{pub_key}/cloud.ply` | PLY 파일 다운로드 |
| GET | `/recon/pub/{pub_key}/scene.splat` | Splat 파일 다운로드 |
| GET | `/v/{pub_key}` | 3D 뷰어 |

### 1. 작업 생성

```bash
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg" \
  -F "original_resolution=false"
```

**응답:**
```json
{
  "job_id": "tOFL7kfe",
  "pub_key": "cWjwdgZjSA",
  "original_resolution": false
}
```

### 2. 작업 상태 확인 (Step & Progress 포함)

```bash
curl http://localhost:8000/recon/jobs/tOFL7kfe/status | jq
```

**응답:**
```json
{
  "job_id": "tOFL7kfe",
  "status": "PROCESSING",
  "step": "GS_TRAIN",
  "progress": 65,
  "log_tail": [
    ">> [COLMAP_FEAT] Extracting features...",
    ">> [GS_TRAIN] Starting Gaussian Splatting training...",
    "Training progress:  45%|████▌     | 4500/10000"
  ],
  "created_at": "2025-10-13T08:00:01.586347",
  "started_at": "2025-10-13T08:00:01.598771"
}
```

**Step 진행 단계:**
- `QUEUED` (0%): 대기 중
- `PREFLIGHT` (5%): 사전 점검
- `COLMAP_FEAT` (15%): 특징점 추출
- `COLMAP_MATCH` (30%): 특징점 매칭
- `COLMAP_MAP` (45%): Sparse 재구성
- `COLMAP_UNDIST` (55%): 이미지 왜곡 보정
- `GS_TRAIN` (65%): Gaussian Splatting 학습
- `EXPORT_PLY` (90%): 후처리 및 내보내기
- `DONE` (100%): 완료
- `ERROR` (0%): 오류 발생

## 개발 가이드

### 코드 스타일

- **Python**: PEP 8 준수
- **Docstring**: Google Style
- **Import 순서**: 표준 라이브러리 → 서드파티 → 로컬
- **함수명**: `snake_case`
- **클래스명**: `PascalCase`
- **상수**: `UPPER_SNAKE_CASE`

### 새 기능 추가 예시

#### 1. 새 API 엔드포인트 추가

```python
# app/api/jobs.py
@router.get("/jobs/{job_id}/metrics")
async def get_job_metrics(job_id: str):
    """작업 메트릭 조회"""
    db = SessionLocal()
    try:
        job = crud.get_job_by_id(db, job_id)
        if not job:
            raise HTTPException(404, "Job not found")

        return {
            "gaussian_count": job.gaussian_count,
            "processing_time": job.processing_time_seconds,
            "colmap_points": job.colmap_points
        }
    finally:
        db.close()
```

#### 2. DB 모델 필드 추가

```python
# 1. app/db/models.py 수정
class Job(Base):
    # ... 기존 필드들
    custom_field = Column(String(100), nullable=True)

# 2. 데이터베이스 마이그레이션
sqlite3 gaussian_splatting.db "ALTER TABLE jobs ADD COLUMN custom_field VARCHAR(100);"

# 3. app/db/crud.py에 update 함수 추가
def update_custom_field(db: Session, job_id: str, value: str):
    job = get_job_by_id(db, job_id)
    if job:
        job.custom_field = value
        db.commit()
    return job
```

#### 3. 새 필터 알고리즘 추가

```python
# app/utils/outlier_filter.py
def custom_filter(ply_path: str, output_path: str, threshold: float = 0.5):
    """커스텀 필터링 알고리즘"""
    plydata = PlyData.read(ply_path)
    vertex = plydata['vertex']

    # 필터링 로직
    positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
    # ... 알고리즘 구현

    # 결과 저장
    filtered_vertex = np.array([...], dtype=vertex.dtype)
    PlyData([Element.describe(filtered_vertex, 'vertex')]).write(output_path)

# app/core/gaussian_splatting.py에서 사용
from app.utils.outlier_filter import custom_filter
custom_filter(input_ply, output_ply, threshold=0.7)
```

### 로깅 추가

```python
from app.utils.logger import setup_logger

logger = setup_logger(__name__)

def my_function():
    logger.info("처리 시작")
    logger.debug(f"상세 정보: {data}")
    logger.warning("경고 메시지")
    logger.error("오류 발생")
```

### 설정 값 추가

```python
# app/config.py
class Settings:
    # 기존 설정들...

    # 새 설정 추가
    NEW_FEATURE_ENABLED: bool = os.getenv("NEW_FEATURE_ENABLED", "False").lower() == "true"
    NEW_THRESHOLD: float = float(os.getenv("NEW_THRESHOLD", "0.5"))

# 사용
from app.config import settings
if settings.NEW_FEATURE_ENABLED:
    # 기능 실행
```

## 브랜치 전략

### 메인 브랜치

- **`master`**: 프로덕션 안정 버전
- **`feature/*`**: 새 기능 개발
- **`bugfix/*`**: 버그 수정
- **`hotfix/*`**: 긴급 수정

### 워크플로우

```bash
# 1. 새 기능 브랜치 생성
git checkout master
git pull origin master
git checkout -b feature/new-feature

# 2. 개발 및 커밋
git add .
git commit -m "feat: Add new feature

Detailed description of changes"

# 3. 푸시
git push origin feature/new-feature

# 4. Pull Request 생성 (GitHub/GitLab)

# 5. 리뷰 후 master에 머지
```

### 커밋 메시지 컨벤션

```
<type>: <subject>

<body>

<footer>
```

**Type:**
- `feat`: 새 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅
- `refactor`: 리팩토링
- `test`: 테스트 추가
- `chore`: 빌드/설정 변경

**예시:**
```
feat: Add progress tracking to pipeline

- Add step and progress fields to Job model
- Update process_job to track 8 stages
- Display progress percentage in API response

Implements IMPLEMENT.md section E
```

## 테스트

### 수동 테스트

```bash
# 1. Health check
curl http://localhost:8000/healthz

# 2. Preflight check
python -c "from app.utils.preflight import run_preflight_check; print(run_preflight_check().get_summary())"

# 3. 작업 생성 및 모니터링
JOB_ID=$(curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@test1.jpg" \
  -F "files=@test2.jpg" \
  -F "files=@test3.jpg" \
  | jq -r '.job_id')

# 4. 상태 확인 (반복)
watch -n 5 "curl -s http://localhost:8000/recon/jobs/$JOB_ID/status | jq '.status, .step, .progress'"
```

### 업로드 검증 테스트

```bash
# 1. 이미지 개수 부족 (< 3장)
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
# 예상: 400 에러, "이미지 3~20장만 허용합니다"

# 2. 파일 크기 초과 (> 30MB)
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@large_file.jpg"
# 예상: 413 에러, "File too large"

# 3. 잘못된 MIME 타입
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@document.pdf"
# 예상: 400 에러, "Unsupported MIME type"
```

### 부하 테스트

```bash
# 대기열 테스트 (동시 3개 작업 생성)
for i in {1..3}; do
  curl -X POST http://localhost:8000/recon/jobs \
    -F "files=@test1.jpg" \
    -F "files=@test2.jpg" \
    -F "files=@test3.jpg" &
done

# 대기열 상태 확인
curl http://localhost:8000/recon/queue | jq
```

## 문제 해결

### COLMAP 실패

**증상:** 작업이 COLMAP 단계에서 실패

**원인:**
- 이미지 품질 부족
- 특징점 부족 (흐린 이미지, 단순한 표면)
- 중복 이미지 또는 각도 변화 부족

**해결:**
```bash
# 1. 이미지 품질 확인
identify -verbose image.jpg | grep -E "Geometry|Format"

# 2. 다양한 각도의 이미지 사용
# 3. 고해상도 이미지 권장 (최소 640x480)
```

### GPU 메모리 부족

**증상:** `CUDA out of memory` 에러

**해결:**
```python
# app/config.py
TRAINING_ITERATIONS = 7000  # 10000에서 감소
MAX_IMAGE_SIZE = 1200       # 1600에서 감소

# 또는
MAX_CONCURRENT_JOBS = 1     # 이미 기본값
```

```bash
# GPU 메모리 확인
nvidia-smi

# 프로세스 강제 종료
lsof -ti:8000 | xargs kill -9
```

### 포트 충돌

**증상:** `Address already in use`

**해결:**
```bash
# 기존 프로세스 종료
lsof -ti:8000 | xargs kill -9

# 또는 포트 변경
# app/config.py
PORT = 8001
```

### 데이터베이스 마이그레이션 오류

**증상:** 새 필드 추가 후 에러

**해결:**
```bash
# SQLite 마이그레이션
sqlite3 gaussian_splatting.db "ALTER TABLE jobs ADD COLUMN new_field VARCHAR(100);"

# 또는 DB 재생성 (주의: 데이터 손실)
rm gaussian_splatting.db
python -c "from app.db.database import init_db; init_db()"
```

### NumPy 버전 호환성

**증상:** `AttributeError: module 'numpy' has no attribute 'xxx'`

**해결:**
```bash
pip install "numpy<2"
```

## 성능 최적화

### GPU 메모리 관리
- 최대 동시 작업 수: 1개 (기본)
- GPU 메모리 모니터링: 20 iteration마다 자동 체크
- 메모리 부족 시 자동 정리

### 처리 시간 예상
- **이미지 3-10장**: 약 5-10분
- **이미지 10-20장**: 약 10-20분

실제 시간은 이미지 해상도, GPU 성능, COLMAP 복잡도에 따라 달라집니다.

## 기여 방법

1. 이 레포지토리를 Fork
2. 새 기능 브랜치 생성 (`git checkout -b feature/AmazingFeature`)
3. 변경사항 커밋 (`git commit -m 'feat: Add AmazingFeature'`)
4. 브랜치에 푸시 (`git push origin feature/AmazingFeature`)
5. Pull Request 생성

## 기술 스택

- **Backend**: FastAPI, Uvicorn, SQLAlchemy
- **3D Processing**: COLMAP, Gaussian Splatting
- **Machine Learning**: PyTorch, scikit-learn
- **Frontend**: Three.js, antimatter15 splat viewer
- **Database**: SQLite (기본), PostgreSQL (선택 가능)
- **Async**: asyncio, aiofiles

## 라이센스

이 프로젝트는 다음 오픈소스 프로젝트를 사용합니다:
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Inria License
- [COLMAP](https://colmap.github.io/) - BSD License
- [antimatter15 splat-viewer](https://github.com/antimatter15/splat) - MIT License

## 참고 자료

- [3D Gaussian Splatting 논문](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP 문서](https://colmap.github.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [SQLAlchemy 문서](https://docs.sqlalchemy.org/)

## 지원

문제가 발생하면 GitHub Issues에 등록해주세요.
