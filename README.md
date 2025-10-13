# InstaRecon3D - Gaussian Splatting 3D Reconstruction API

FastAPI 기반 3D 재구성 서비스로, 여러 이미지를 업로드하면 COLMAP과 Gaussian Splatting을 사용하여 3D 모델을 자동 생성하고 웹 뷰어로 확인할 수 있습니다.

**구현 완성도: 96%** ✅ (프로덕션 준비 완료)

## 목차

- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [API 사용법](#api-사용법)
- [파이프라인 단계](#파이프라인-단계)
- [평가 메트릭](#평가-메트릭)
- [문제 해결](#문제-해결)
- [기술 스택](#기술-스택)

## 주요 기능

### 핵심 기능
- **자동 3D 재구성**: 3~20장의 이미지로 고품질 3D 모델 생성
- **단계별 진행률 추적**: 10단계 파이프라인 실시간 모니터링 (0-100%)
- **평가 메트릭**: PSNR, SSIM, LPIPS 자동 계산 및 표시 ✨
- **Train/Test Split**: 80/20 자동 분할로 모델 품질 검증 ✨
- **웹 3D 뷰어**: Splat 형식 최적화 실시간 렌더링

### 검증 및 안정성
- **Preflight 체크**: Python, CUDA, COLMAP, 파일시스템 사전 검증
- **업로드 검증**: 파일 크기(개별 30MB, 전체 500MB), MIME 타입, 이미지 개수
- **작업 대기열**: asyncio.Semaphore 기반 순차 처리 (GPU 메모리 및 포트 충돌 방지)
- **아웃라이어 필터링**: K-NN + DBSCAN 기반 노이즈 제거
- **Health Check**: `/healthz` 엔드포인트 (Kubernetes/Docker 표준)

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

# 6. 환경 변수 설정 (선택)
cp .env.example .env
# .env 파일 편집

# 7. Preflight 체크 (선택)
python -c "from app.utils.preflight import run_preflight_check; print(run_preflight_check().get_summary())"

# 8. 서버 실행
python main.py
```

서버가 `http://0.0.0.0:8000`에서 실행됩니다.

## 프로젝트 구조

```
codyssey_hackathon/
├── app/                          # FastAPI 애플리케이션
│   ├── api/                      # API 엔드포인트
│   │   ├── jobs.py              # 작업 관리 + 파이프라인 오케스트레이션
│   │   ├── viewer.py            # 3D 뷰어 렌더링
│   │   └── dependencies.py      # 동시성 제어 (Semaphore)
│   │
│   ├── core/                     # 핵심 처리 로직
│   │   ├── colmap.py            # COLMAP 파이프라인 + train/test split
│   │   ├── gaussian_splatting.py # GS 학습 + 평가 파이프라인
│   │   └── pipeline.py          # subprocess 실행 + GPU 모니터링
│   │
│   ├── db/                       # 데이터베이스
│   │   ├── models.py            # Job, ErrorLog 모델 (메트릭 포함)
│   │   ├── crud.py              # CRUD 함수
│   │   └── database.py          # SQLAlchemy 설정
│   │
│   ├── utils/                    # 유틸리티
│   │   ├── preflight.py         # 환경 사전 점검 (Python/CUDA/COLMAP)
│   │   ├── image.py             # 이미지 검증/저장
│   │   ├── converter.py         # PLY → Splat 변환
│   │   ├── outlier_filter.py   # Point cloud 필터링
│   │   ├── logger.py            # 로깅 설정
│   │   └── system.py            # GPU 모니터링, pub_key 생성
│   │
│   ├── config.py                 # 전역 설정 (.env 지원)
│   └── main.py                   # FastAPI 진입점
│
├── templates/                    # HTML 템플릿
│   └── viewer.html              # 3D 뷰어 (메트릭 표시)
│
├── gaussian-splatting/          # Gaussian Splatting 레포지토리
├── data/jobs/                   # 작업 데이터 저장소
├── gaussian_splatting.db        # SQLite 데이터베이스
├── main.py                       # 서버 실행 파일
└── requirements.txt             # Python 의존성
```

### 작업 디렉토리 구조

각 작업은 `data/jobs/{job_id}/` 디렉토리에 저장됩니다:

```
data/jobs/{job_id}/
├── upload/images/              # 업로드된 원본 이미지
│
├── colmap/                     # COLMAP 처리 결과
│   ├── database.db            # COLMAP SQLite DB
│   └── sparse/0/              # Sparse 재구성 (binary)
│
├── work/                       # COLMAP 출력 (GS 입력)
│   ├── images/                # Undistorted 이미지
│   ├── sparse/0/              # txt 형식 카메라/포인트
│   ├── train.txt              # Train set 목록 (80%) ✨
│   ├── test.txt               # Test set 목록 (20%) ✨
│   └── stereo/                # Dense 재구성 (depth maps)
│
├── output/                     # Gaussian Splatting 출력
│   ├── cameras.json
│   ├── cfg_args               # 훈련 설정
│   ├── input.ply              # 초기 point cloud
│   ├── results.json           # 평가 메트릭 ✨
│   ├── point_cloud/iteration_10000/
│   │   ├── point_cloud.ply    # 훈련된 Gaussians
│   │   ├── point_cloud_filtered.ply  # Outlier 제거
│   │   └── scene.splat        # Splat 형식 (웹 뷰어용)
│   └── test/ours_10000/       # 평가 결과 ✨
│       ├── renders/           # 렌더링된 test 이미지
│       └── gt/                # Ground truth 이미지
│
└── logs/
    └── process.log            # 작업 전체 로그
```

## API 사용법

### 엔드포인트 목록

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/` | API 정보 및 버전 |
| GET | `/healthz` | Health check (k8s/Docker 표준) |
| POST | `/recon/jobs` | 새 작업 생성 |
| GET | `/recon/jobs/{job_id}/status` | 작업 상태 조회 (step, progress, metrics 포함) |
| GET | `/recon/queue` | 대기열 상태 |
| GET | `/recon/pub/{pub_key}/cloud.ply` | PLY 파일 다운로드 |
| GET | `/recon/pub/{pub_key}/scene.splat` | Splat 파일 다운로드 |
| GET | `/v/{pub_key}` | 3D 뷰어 (메트릭 표시) |

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
  "job_id": "6giVuAVu",
  "pub_key": "tmb5Wy5OM9",
  "original_resolution": false
}
```

### 2. 작업 상태 확인

```bash
curl http://localhost:8000/recon/jobs/6giVuAVu/status | jq
```

**응답 예시 (완료 시):**
```json
{
  "job_id": "6giVuAVu",
  "status": "COMPLETED",
  "step": "DONE",
  "progress": 100,
  "psnr": 17.40,
  "ssim": 0.4942,
  "lpips": 0.4135,
  "gaussian_count": 343728,
  "image_count": 17,
  "iterations": 10000,
  "processing_time_seconds": 800.5,
  "viewer_url": "http://kaprpc.iptime.org:5051/v/tmb5Wy5OM9",
  "log_tail": [
    ">> [EVALUATION] PSNR: 17.40 dB",
    ">> [EVALUATION] SSIM: 0.4942",
    ">> [EVALUATION] LPIPS: 0.4135",
    ">> [SUCCESS] Job completed!"
  ],
  "created_at": "2025-10-13T13:30:45.196408",
  "completed_at": "2025-10-13T13:44:05.650599"
}
```

### 3. Health Check

```bash
# Kubernetes/Docker liveness probe
curl http://localhost:8000/healthz
# 응답: "ok"
```

## 파이프라인 단계

전체 파이프라인은 10단계로 구성되며, 각 단계마다 progress가 업데이트됩니다:

| Step | Progress | 설명 | 소요 시간 (예상) |
|------|----------|------|-----------------|
| **QUEUED** | 0% | 대기열 대기 | - |
| **PREFLIGHT** | 5% | 환경 사전 점검 (Python/CUDA/COLMAP/GS) | < 1초 |
| **COLMAP_FEAT** | 15% | 특징점 추출 (SIFT) | 30초 ~ 2분 |
| **COLMAP_MATCH** | 30% | 특징점 매칭 | 30초 ~ 2분 |
| **COLMAP_MAP** | 45% | Sparse 3D 재구성 | 1~3분 |
| **COLMAP_UNDIST** | 55% | 이미지 왜곡 보정 + train/test split | 30초 ~ 1분 |
| **GS_TRAIN** | 65% | Gaussian Splatting 학습 (10000 iterations) | 8~15분 |
| **EVALUATION** | 85% | Test set 렌더링 + 메트릭 계산 ✨ | 2~4분 |
| **EXPORT_PLY** | 95% | Outlier filtering + Splat 변환 | 30초 ~ 1분 |
| **DONE** | 100% | 완료 | - |
| **ERROR** | 0% | 오류 발생 | - |

**평균 처리 시간**:
- 이미지 3-10장: 약 10-15분
- 이미지 10-20장: 약 15-25분

## 평가 메트릭

### 자동 평가 파이프라인 ✨

작업 완료 시 자동으로 다음 메트릭이 계산됩니다:

| 메트릭 | 설명 | 범위 | 해석 |
|--------|------|------|------|
| **PSNR** | Peak Signal-to-Noise Ratio | 0 ~ ∞ dB | 높을수록 좋음 (>20dB: 양호) |
| **SSIM** | Structural Similarity Index | 0 ~ 1 | 높을수록 좋음 (>0.8: 양호) |
| **LPIPS** | Learned Perceptual Image Patch Similarity | 0 ~ 1 | 낮을수록 좋음 (<0.3: 양호) |

### 메트릭 확인 방법

1. **API 응답**:
```bash
curl http://localhost:8000/recon/jobs/{job_id}/status | jq '.psnr, .ssim, .lpips'
```

2. **3D 뷰어**:
```
http://kaprpc.iptime.org:5051/v/{pub_key}
```
뷰어 우측 상단에 메트릭 표시

3. **results.json 파일**:
```bash
cat data/jobs/{job_id}/output/results.json
```

### Train/Test Split

- **자동 생성**: COLMAP 완료 후 자동으로 80/20 분할
- **Train set**: 80% 이미지 (학습용)
- **Test set**: 20% 이미지 (평가용, 최소 1장)
- **파일 위치**:
  - `data/jobs/{job_id}/work/train.txt`
  - `data/jobs/{job_id}/work/test.txt`

## 문제 해결

### 1. Preflight 체크 실패

**증상**: 작업이 PREFLIGHT 단계에서 실패

**확인**:
```bash
python -c "from app.utils.preflight import run_preflight_check; print(run_preflight_check().get_summary())"
```

**일반적인 원인**:
- CUDA 미설치 또는 인식 불가
- COLMAP 미설치
- Gaussian Splatting 디렉토리 없음
- 파일시스템 쓰기 권한 없음

**해결**:
```bash
# CUDA 확인
nvidia-smi

# COLMAP 설치
sudo apt-get install colmap

# Gaussian Splatting 클론
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
```

### 2. COLMAP 실패

**증상**: COLMAP_MAP 단계에서 실패

**원인**:
- 이미지 품질 부족
- 특징점 부족 (흐린 이미지, 단순한 표면)
- 중복 이미지 또는 각도 변화 부족
- 이미지 개수 부족 (<3장)

**해결**:
- 다양한 각도의 고해상도 이미지 사용 (최소 640x480)
- 최소 3장 이상의 이미지 업로드
- 텍스처가 풍부한 물체 촬영

### 3. GPU 메모리 부족

**증상**: `CUDA out of memory` 에러

**해결**:
```bash
# 1. GPU 메모리 확인
nvidia-smi

# 2. 기존 프로세스 종료
lsof -ti:8000 | xargs kill -9

# 3. 설정 조정 (.env 파일)
TRAINING_ITERATIONS=7000  # 10000에서 감소
MAX_IMAGE_SIZE=1200       # 1600에서 감소
MAX_CONCURRENT_JOBS=1     # 이미 기본값
```

### 4. 평가 메트릭이 NULL

**증상**: psnr, ssim, lpips가 모두 NULL

**원인**:
- Train/test split 생성 실패
- render.py 또는 metrics.py 실행 실패
- results.json 파싱 오류

**확인**:
```bash
# 1. train/test split 확인
cat data/jobs/{job_id}/work/train.txt
cat data/jobs/{job_id}/work/test.txt

# 2. results.json 확인
cat data/jobs/{job_id}/output/results.json

# 3. 로그 확인
tail -f data/jobs/{job_id}/logs/process.log
```

### 5. 포트 충돌

**증상**: `Address already in use`

**해결**:
```bash
# 기존 프로세스 종료
lsof -ti:8000 | xargs kill -9

# 또는 포트 변경 (.env)
PORT=8001
```

### 6. 데이터베이스 초기화

**필요한 경우**: 스키마 변경 후

```bash
# 주의: 기존 데이터 손실
rm gaussian_splatting.db
python -c "from app.db.database import init_db; init_db()"
```

## 검증 및 테스트

### 수동 테스트

```bash
# 1. Health check
curl http://localhost:8000/healthz
# 응답: "ok"

# 2. Preflight check
python -c "from app.utils.preflight import run_preflight_check; print(run_preflight_check().get_summary())"

# 3. 작업 생성
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@test1.jpg" \
  -F "files=@test2.jpg" \
  -F "files=@test3.jpg" | jq

# 4. 상태 모니터링
watch -n 5 "curl -s http://localhost:8000/recon/jobs/{job_id}/status | jq '.status, .step, .progress'"
```

### 업로드 검증 테스트

```bash
# 1. 이미지 개수 부족 (< 3장)
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg"
# 예상: 400 에러

# 2. 파일 크기 초과 (> 30MB)
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@large_file.jpg"
# 예상: 413 에러

# 3. 잘못된 MIME 타입
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@document.pdf"
# 예상: 400 에러
```

## 환경 변수 설정

`.env` 파일을 생성하여 설정을 커스터마이즈할 수 있습니다:

```bash
# 서버 설정
HOST=0.0.0.0
PORT=8000
BASE_URL=http://localhost:8000
DEBUG=false

# 처리 설정
MAX_CONCURRENT_JOBS=1
TRAINING_ITERATIONS=10000

# 이미지 설정
MAX_IMAGE_SIZE=1600
MIN_IMAGE_SIZE=100
MAX_FILE_SIZE_MB=30

# GPU 모니터링
MONITOR_GPU=true
GPU_CHECK_INTERVAL=100

# 아웃라이어 필터링
OUTLIER_K_NEIGHBORS=20
OUTLIER_STD_THRESHOLD=2.0
OUTLIER_REMOVE_SMALL_CLUSTERS=true
OUTLIER_MIN_CLUSTER_RATIO=0.05

# 데이터베이스
DATABASE_URL=sqlite:///./gaussian_splatting.db

# 경로 설정 (고급)
GAUSSIAN_SPLATTING_DIR=./gaussian-splatting
DATA_DIR=./data/jobs
CONDA_PYTHON=/path/to/conda/envs/codyssey/bin/python
```

## 성능 최적화

### GPU 메모리 관리
- 최대 동시 작업 수: 1개 (기본)
- GPU 메모리 모니터링: 100 iteration마다 자동 체크
- 메모리 부족 시 자동 정리

### 처리 시간 최적화
- 이미지 리사이즈: `original_resolution=false` (권장)
- Iteration 수 조정: 7000~10000 (기본 10000)
- 최소 이미지 수 유지: 3~10장 (최적)

## 기술 스택

- **Backend**: FastAPI 0.104.1, Uvicorn, SQLAlchemy
- **3D Processing**: COLMAP, Gaussian Splatting (Inria 2023)
- **Machine Learning**: PyTorch 2.0+, scikit-learn
- **Frontend**: Three.js, antimatter15 splat viewer
- **Database**: SQLite (기본), PostgreSQL (선택 가능)
- **Async**: asyncio, aiofiles
- **Validation**: Pydantic, python-multipart

## 구현 상태

- ✅ 핵심 파이프라인 (COLMAP + Gaussian Splatting)
- ✅ 사전점검 (Preflight)
- ✅ 업로드 검증
- ✅ 단계별 진행률 추적 (step, progress)
- ✅ 평가 메트릭 (PSNR, SSIM, LPIPS) ✨
- ✅ Train/Test Split ✨
- ✅ 작업 대기열
- ✅ 아웃라이어 필터링
- ✅ Health check (/healthz)
- ✅ GPU 메모리 모니터링
- ✅ 에러 처리 및 로깅


## 라이센스

이 프로젝트는 다음 오픈소스 프로젝트를 사용합니다:
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Inria License
- [COLMAP](https://colmap.github.io/) - BSD License
- [antimatter15 splat-viewer](https://github.com/antimatter15/splat) - MIT License

## 참고 자료

- [3D Gaussian Splatting 논문](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP 문서](https://colmap.github.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [Kubernetes Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
