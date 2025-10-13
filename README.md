# Gaussian Splatting 3D Reconstruction API

FastAPI 기반 3D 재구성 서비스로, 여러 이미지를 업로드하면 COLMAP과 Gaussian Splatting을 사용하여 3D 모델을 생성하고 웹 뷰어로 확인할 수 있습니다.

## 주요 기능

- **이미지 업로드 & 3D 재구성**: 여러 이미지로 3D 모델 자동 생성
- **작업 대기열 시스템**: asyncio.Semaphore 기반 순차 처리 (포트 충돌 방지)
- **실시간 상태 추적**: 작업 진행률, 대기열 위치, GPU 메모리 모니터링
- **아웃라이어 필터링**: K-NN + DBSCAN 기반 노이즈 제거
- **웹 3D 뷰어**: Splat 형식으로 최적화된 실시간 렌더링
- **데이터베이스 관리**: SQLAlchemy ORM 기반 작업 이력 관리

## 시스템 아키텍처

### 전체 처리 흐름

```
사용자 이미지 업로드
    ↓
FastAPI 작업 생성 (POST /recon/jobs)
    ↓
asyncio.Semaphore 대기열 진입
    ↓
[COLMAP Pipeline]
├─ Feature Extraction (SIFT)
├─ Feature Matching
├─ Sparse Reconstruction
└─ Image Undistortion
    ↓
[Gaussian Splatting Training]
└─ 10,000 iterations (약 10-15분)
    ↓
[Post-Processing]
├─ K-NN Outlier Filtering
├─ DBSCAN Cluster Filtering
└─ PLY → Splat 변환
    ↓
웹 뷰어로 3D 모델 확인
```

### 프로젝트 구조

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
├── templates/                    # HTML 템플릿
│   └── viewer.html              # 3D 뷰어 페이지
├── logs/                         # 서버 로그
├── gaussian-splatting/          # Gaussian Splatting 레포지토리
├── data/                         # 작업 데이터 저장소
│   └── jobs/{job_id}/           # 개별 작업 디렉토리
│       ├── upload/images/       # 업로드된 원본 이미지
│       ├── colmap/              # COLMAP 처리 결과
│       ├── work/                # 중간 처리 데이터
│       ├── output/              # 최종 결과물
│       │   └── point_cloud/
│       │       └── iteration_10000/
│       │           ├── point_cloud.ply          # 원본 포인트 클라우드
│       │           ├── point_cloud_filtered.ply # 필터링된 결과
│       │           └── scene.splat              # 웹 뷰어용 최적화 파일
│       └── logs/
│           └── process.log      # 작업 처리 로그
└── gaussian_splatting.db        # SQLite 데이터베이스
```

## 핵심 기능 상세

### 1. 작업 대기열 시스템

```python
# app/api/jobs.py
job_semaphore = asyncio.Semaphore(settings.MAX_CONCURRENT_JOBS)  # MAX = 1

async def process_job(job_id: str, original_resolution: bool):
    async with job_semaphore:  # 세마포어 획득 시까지 대기
        # 실제 처리 로직
        crud.update_job_status(db, job_id, "PROCESSING")
        # COLMAP + Gaussian Splatting
```

**특징:**
- 동시 실행 제한으로 GPU 메모리 및 포트 충돌 방지
- FIFO 순서로 자동 처리
- 대기열 위치 실시간 표시

### 2. 아웃라이어 필터링

**Step 1: K-Nearest Neighbors**
```python
# app/utils/outlier_filter.py
mean_distances = knn_distances.mean(axis=1)
threshold = mean_dist + std_threshold * std_dist
inlier_mask = mean_distances < threshold
```

**Step 2: DBSCAN Clustering**
```python
clustering = DBSCAN(eps=mean_dist*3, min_samples=10).fit(positions)
# 작은 클러스터 제거 (min_cluster_ratio=0.01)
```

### 3. PLY → Splat 변환

```python
# app/utils/converter.py
def convert_ply_to_splat(ply_path, splat_path):
    # 각 Gaussian을 44 bytes로 압축
    # - Position: 3 floats (12 bytes)
    # - Scale: 3 floats (12 bytes)
    # - Color + Opacity: 4 bytes
    # - Rotation (Quaternion): 4 floats (16 bytes)
```

**최적화:**
- SH coefficients → RGB 변환
- Log scale → Linear scale
- Quaternion 정규화
- 바이너리 포맷으로 빠른 로딩

### 4. 데이터베이스 스키마

```python
# app/db/models.py
class Job(Base):
    job_id: str              # 8자 영숫자 ID
    pub_key: str             # 10자 public key (뷰어 접근용)
    status: str              # PENDING, PROCESSING, COMPLETED, FAILED
    image_count: int
    gaussian_count: int      # 생성된 Gaussian 개수
    created_at: datetime
    started_at: datetime
    completed_at: datetime
    error_message: str
```

## 설치 방법

### 1. 환경 설정

```bash
# Conda 환경 생성
conda create -n codyssey python=3.9
conda activate codyssey

# PyTorch 설치 (CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# COLMAP 설치 (Ubuntu)
sudo apt-get install colmap

# scikit-learn 설치 (아웃라이어 필터링용)
pip install scikit-learn
```

### 2. Gaussian Splatting 설정

```bash
# 프로젝트 루트에서
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

# 서브모듈 빌드
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn
cd ..
```

### 3. 프로젝트 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# 디렉토리 구조 확인 (자동 생성됨)
python main.py
```

## 실행 방법

```bash
# 서버 시작
python main.py

# 또는 직접 실행
python3 -m app.main
```

서버가 http://0.0.0.0:8000 에서 실행됩니다.

## API 사용법

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
  "job_id": "iOKlxSgF",
  "pub_key": "CzWrYDrbd9",
  "original_resolution": false
}
```

### 2. 작업 상태 확인

```bash
curl http://localhost:8000/recon/jobs/iOKlxSgF/status | jq
```

**응답 (대기 중):**
```json
{
  "job_id": "iOKlxSgF",
  "status": "PENDING",
  "queue_position": 1,
  "log_tail": [
    ">> [QUEUE] Position in queue: 1",
    ">> [QUEUE] Currently running: 1/1 jobs",
    ">> [QUEUE] Waiting for processing slot..."
  ]
}
```

**응답 (처리 중):**
```json
{
  "job_id": "iOKlxSgF",
  "status": "PROCESSING",
  "log_tail": [
    "Training progress: 45%|████▌     | 4500/10000 [03:25<04:10, 21.97it/s]",
    "[GPU Memory] GPU 0: 1473MB / 16380MB (9.0%)"
  ]
}
```

**응답 (완료):**
```json
{
  "job_id": "iOKlxSgF",
  "status": "COMPLETED",
  "gaussian_count": 220695,
  "viewer_url": "http://kaprpc.iptime.org:5051/v/CzWrYDrbd9",
  "processing_time_seconds": 847.3
}
```

### 3. 대기열 상태 확인

```bash
curl http://localhost:8000/recon/queue | jq
```

**응답:**
```json
{
  "max_concurrent": 1,
  "running_count": 1,
  "pending_count": 2,
  "running_jobs": [
    {
      "job_id": "iOKlxSgF",
      "created_at": "2025-10-13T02:49:32.019182"
    }
  ],
  "pending_jobs": [
    {
      "job_id": "j5alVPQN",
      "position": 1,
      "created_at": "2025-10-13T02:49:50.812597"
    },
    {
      "job_id": "o2GCi0NS",
      "position": 2,
      "created_at": "2025-10-13T03:05:49.829802"
    }
  ]
}
```

### 4. 결과 다운로드

```bash
# PLY 파일 (원본 포인트 클라우드)
curl -O http://localhost:8000/recon/pub/CzWrYDrbd9/cloud.ply

# Splat 파일 (웹 뷰어용)
curl -O http://localhost:8000/recon/pub/CzWrYDrbd9/scene.splat
```

### 5. 웹 뷰어 접근

브라우저에서 다음 URL 접속:
```
http://localhost:8000/v/CzWrYDrbd9
```

**마우스 컨트롤:**
- 좌클릭 + 드래그: 회전
- 우클릭 + 드래그: 이동
- 스크롤: 확대/축소

## 설정 파일

### app/config.py

주요 설정 값들:

```python
# 서버 설정
HOST = "0.0.0.0"
PORT = 8000
BASE_URL = "http://kaprpc.iptime.org:5051"

# 동시 작업 제한
MAX_CONCURRENT_JOBS = 1

# 이미지 제한
MIN_IMAGES = 3
MAX_IMAGES = 200
MAX_IMAGE_SIZE = 1600

# Gaussian Splatting 학습
TRAINING_ITERATIONS = 10000
DENSIFY_UNTIL_ITER = 5000

# COLMAP 설정
COLMAP_MAX_FEATURES = 16384
COLMAP_NUM_THREADS = 8

# 아웃라이어 필터링
OUTLIER_K_NEIGHBORS = 20
OUTLIER_STD_THRESHOLD = 2.0
OUTLIER_REMOVE_SMALL_CLUSTERS = True
OUTLIER_MIN_CLUSTER_RATIO = 0.01
```

## 성능 최적화

### GPU 메모리 관리
- 최대 동시 작업 수: 1개 (포트 충돌 및 메모리 관리)
- GPU 메모리 모니터링: 20 iteration마다 자동 체크
- 작업별 격리된 프로세스

### 디스크 공간 관리
- 필터링 전후 PLY 파일 모두 보관
- Splat 파일: 원본 대비 약 50-70% 크기
- 로그 파일: 자동 rotation 없음 (수동 관리 필요)

### 처리 시간 예상
- **이미지 3-10장**: 약 5-10분
- **이미지 10-50장**: 약 10-20분
- **이미지 50-200장**: 약 20-40분

시간은 이미지 해상도, GPU 성능, COLMAP 복잡도에 따라 달라집니다.

## 문제 해결

### 1. COLMAP 실패
```bash
# 원인: 이미지 품질 부족, 중복 이미지, 특징점 부족
# 해결: 다양한 각도의 고품질 이미지 사용
```

### 2. GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# config.py에서 iteration 수 감소
TRAINING_ITERATIONS = 7000
```

### 3. Port already in use
```bash
# 기존 프로세스 종료
lsof -ti:8000 | xargs kill -9

# 또는 포트 변경
# app/config.py의 PORT 수정
```

### 4. NumPy 버전 오류
```bash
pip install "numpy<2"
```

## 개발 가이드

### 새로운 필터 추가

```python
# app/utils/outlier_filter.py
def custom_filter(ply_path, output_path, **params):
    plydata = PlyData.read(ply_path)
    # 필터링 로직
    # ...
    PlyData([filtered_element]).write(output_path)

# app/core/gaussian_splatting.py에서 호출
from app.utils.outlier_filter import custom_filter
custom_filter(ply_file, output_file, param1=value1)
```

### API 엔드포인트 추가

```python
# app/api/jobs.py
@router.get("/custom-endpoint")
async def custom_endpoint():
    # 로직
    return {"result": "data"}
```

### 데이터베이스 마이그레이션

현재는 SQLite 사용, 필요시 PostgreSQL로 전환:

```python
# app/config.py
DATABASE_URL = "postgresql://user:pass@localhost/dbname"

# 마이그레이션 도구 사용 (Alembic 등)
```

## 기술 스택

- **Backend**: FastAPI, Uvicorn, SQLAlchemy
- **3D Processing**: COLMAP, Gaussian Splatting
- **Machine Learning**: PyTorch, scikit-learn
- **Frontend**: Three.js, antimatter15 splat viewer
- **Database**: SQLite (기본), PostgreSQL (선택)
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
