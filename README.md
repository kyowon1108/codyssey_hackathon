# InstaRecon3D - Gaussian Splatting 3D Reconstruction API

FastAPI 기반 3D 재구성 서비스로, 여러 이미지를 업로드하면 COLMAP과 Gaussian Splatting을 사용하여 3D 모델을 자동 생성하고 웹 뷰어로 확인할 수 있습니다.

## 목차

- [주요 기능](#주요-기능)
- [시스템 요구사항](#시스템-요구사항)
- [빠른 시작](#빠른-시작)
- [프로젝트 구조](#프로젝트-구조)
- [API 사용법](#api-사용법)
- [파이프라인 단계](#파이프라인-단계)
- [3D 뷰어](#3d-뷰어)
- [평가 메트릭](#평가-메트릭)
- [문제 해결](#문제-해결)
- [기술 스택](#기술-스택)

## 주요 기능

### 핵심 기능
- **자동 3D 재구성**: 3~20장의 이미지로 고품질 3D 모델 생성
- **단계별 진행률 추적**: 10단계 파이프라인 실시간 모니터링 (0-100%)
- **평가 메트릭**: PSNR, SSIM, LPIPS 자동 계산 및 표시
- **Train/Test Split**: 80/20 자동 분할로 모델 품질 검증
- **PlayCanvas 3D 뷰어**: 공식 Model Viewer로 Gaussian Splatting PLY 실시간 렌더링

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
- **Node.js**: 18.0+ (PlayCanvas 뷰어 빌드용, 선택사항)

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
│   │   └── viewer.py            # 3D 뷰어 리다이렉션
│   │
│   ├── core/                     # 핵심 처리 로직
│   │   ├── colmap.py            # COLMAP 파이프라인 + train/test split
│   │   └── gaussian_splatting.py # GS 학습 + 평가 파이프라인
│   │
│   ├── db/                       # 데이터베이스
│   │   ├── models.py            # Job, ErrorLog 모델 (메트릭 포함)
│   │   ├── crud.py              # CRUD 함수
│   │   └── database.py          # SQLAlchemy 설정
│   │
│   ├── utils/                    # 유틸리티
│   │   ├── preflight.py         # 환경 사전 점검
│   │   ├── image.py             # 이미지 검증/저장
│   │   ├── outlier_filter.py   # Point cloud 필터링
│   │   ├── logger.py            # 로깅 설정
│   │   └── system.py            # GPU 모니터링
│   │
│   ├── config.py                 # 전역 설정 (.env 지원)
│   └── main.py                   # FastAPI 진입점
│
├── viewer/                       # PlayCanvas Model Viewer
│   ├── index.html               # 뷰어 진입점
│   ├── index.js                 # 뷰어 로직 (번들)
│   ├── style.css                # 뷰어 스타일
│   └── static/                  # 아이콘, skybox, 라이브러리
│
├── gaussian-splatting/          # Gaussian Splatting 레포지토리
├── data/                        # 작업 데이터 저장소
│   ├── jobs.db                 # SQLite 데이터베이스 (작업 메타데이터)
│   └── jobs/                   # 작업별 디렉토리
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
│   ├── train.txt              # Train set 목록 (80%)
│   ├── test.txt               # Test set 목록 (20%)
│   └── stereo/                # Dense 재구성 (depth maps)
│
├── output/                     # Gaussian Splatting 출력
│   ├── cameras.json
│   ├── cfg_args               # 훈련 설정
│   ├── input.ply              # 초기 point cloud
│   ├── results.json           # 평가 메트릭
│   ├── exposure.json          # 노출 설정 (평가용)
│   ├── per_view.json          # View별 메트릭
│   ├── point_cloud/iteration_10000/
│   │   ├── point_cloud.ply    # 훈련된 Gaussians
│   │   └── point_cloud_filtered.ply  # Outlier 제거 후
│   └── test/ours_10000/       # 평가 결과
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
| GET | `/recon/pub/{pub_key}/scene.splat` | Splat 파일 다운로드 (legacy, 미사용) |
| GET | `/v/{pub_key}` | 3D 뷰어 (PlayCanvas로 리다이렉트) |
| GET | `/viewer/` | PlayCanvas Model Viewer 직접 접근 |

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
  "psnr": 22.58,
  "ssim": 0.854,
  "lpips": 0.300,
  "gaussian_count": 70636,
  "image_count": 10,
  "iterations": 10000,
  "processing_time_seconds": 800.5,
  "viewer_url": "http://localhost:8000/v/tmb5Wy5OM9",
  "log_tail": [
    ">> [EVALUATION] PSNR: 22.58 dB",
    ">> [EVALUATION] SSIM: 0.8540",
    ">> [EVALUATION] LPIPS: 0.3000",
    ">> [SUCCESS] Job completed! Generated 70636 Gaussians"
  ],
  "created_at": "2025-10-18T00:59:54.218990",
  "completed_at": "2025-10-18T01:13:25.650599"
}
```

### 3. 3D 뷰어 접근

```bash
# 방법 1: Public key로 리다이렉트
http://localhost:8000/v/tmb5Wy5OM9

# 방법 2: 직접 PLY 파일 URL 전달
http://localhost:8000/viewer/?load=http://localhost:8000/recon/pub/tmb5Wy5OM9/cloud.ply
```

### 4. Health Check

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
| **COLMAP_VALIDATE** | 60% | 재구성 품질 검증 (등록률, 3D 포인트 수 등) | < 1초 |
| **GS_TRAIN** | 65% | Gaussian Splatting 학습 (10000 iterations) | 8~15분 |
| **EVALUATION** | 85% | Test set 렌더링 + 메트릭 계산 | 2~4분 |
| **EXPORT_PLY** | 95% | Outlier filtering (노이즈 제거) | 30초 ~ 1분 |
| **DONE** | 100% | 완료 | - |
| **ERROR** | 0% | 오류 발생 | - |

**평균 처리 시간**:
- 이미지 3-10장: 약 10-15분
- 이미지 10-20장: 약 15-25분

## 3D 뷰어

### PlayCanvas Model Viewer

공식 PlayCanvas Model Viewer를 사용하여 Gaussian Splatting 결과를 웹에서 실시간으로 확인할 수 있습니다.

**주요 기능**:
- ✅ Gaussian Splatting PLY 파일 네이티브 지원
- ✅ glTF 2.0 형식도 지원
- ✅ WebGL 및 WebGPU 렌더링
- ✅ 직관적인 카메라 컨트롤 (Orbit/Fly 모드)
- ✅ 다양한 스카이박스 프리셋
- ✅ 드래그 앤 드롭으로 추가 모델 로드 가능
- ✅ 라이팅 및 환경 설정

**접근 방법**:
1. **Public key 사용**: `/v/{pub_key}` - 자동으로 PLY 파일 로드
2. **직접 URL**: `/viewer/?load={ply_url}` - 수동으로 파일 URL 지정
3. **드래그 앤 드롭**: 뷰어에 직접 PLY/glTF 파일 드래그

**컨트롤**:
- 좌클릭 + 드래그: 카메라 회전
- 우클릭 + 드래그: 카메라 이동 (Pan)
- 마우스 휠: 줌 인/아웃
- F 키: 모델에 카메라 포커스

**뷰어 설정**:
- 상단 툴바에서 Orbit/Fly 모드 전환
- 스카이박스 변경 (17개 프리셋)
- 라이팅 강도 조절
- 배경 이미지 업로드 (HDR, PNG, JPG)

## 평가 메트릭

### 자동 평가 파이프라인

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

2. **results.json 파일**:
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

### 2-1. COLMAP 품질 검증 실패

**증상**: COLMAP_VALIDATE 단계에서 실패

**원인 및 해결**:

| 검증 규칙 | 기준 (Error) | 권장 (Warning) | 실패 원인 | 해결 방법 |
|----------|-------------|---------------|----------|----------|
| **등록된 이미지 수** | 최소 3장 | 5장 이상 | 특징점 매칭 실패 | 텍스처가 풍부한 물체 촬영 |
| **이미지 등록률** | 60% 이상 | 80% 이상 | 일부 이미지만 재구성 성공 | 중복/흐린 이미지 제거 |
| **3D 포인트 수** | 최소 300개 | 800개 이상 | 재구성 품질 낮음 | 다양한 각도에서 촬영 |
| **평균 트랙 길이** | 2.5 이상 | 3.5 이상 | 특징점이 적은 뷰에서만 관찰 | 오버랩이 충분한 이미지 사용 |
| **포인트/이미지 비율** | 80 이상 | 100 이상 | 재구성이 너무 sparse | 이미지 품질/개수 증가 |

**기준 설정 근거**:
- Good quality (PSNR >= 20): 최소 10 reg imgs, 822+ points, track 4.04+
- Medium quality (PSNR 15-20): 최소 3 reg imgs, 300+ points, track 2.5+
- 실제 작업 데이터 분석을 바탕으로 품질이 낮은 재구성을 사전에 필터링

**검증 로그 예시**:
```
>> [COLMAP_VALIDATION] Reconstruction Quality Check
============================================================
Statistics:
  - Cameras: 1
  - Total images: 10
  - Registered images: 8
  - 3D points: 1543
  - Avg track length: 3.24

Warnings:
  ⚠ Moderate image registration rate: 80.0% (8/10 images registered)

✓ Reconstruction is valid for training
============================================================
```

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

### 6. 뷰어가 로드되지 않음

**증상**: `/v/{pub_key}` 접근 시 빈 화면

**원인**:
- PLY 파일이 생성되지 않음
- viewer 디렉토리가 없음
- BASE_URL 설정 오류

**확인**:
```bash
# 1. PLY 파일 확인
ls -la data/jobs/{job_id}/output/point_cloud/iteration_10000/

# 2. viewer 디렉토리 확인
ls -la viewer/

# 3. 브라우저 콘솔에서 에러 확인
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

# COLMAP 설정
COLMAP_MAX_FEATURES=8192
COLMAP_NUM_THREADS=8

# 업로드 제한
MIN_IMAGES=3
MAX_IMAGES=20
MAX_TOTAL_SIZE_MB=500
MAX_FILE_SIZE_MB=30

# 타임아웃 (초)
TIMEOUT_COLMAP_SEC=900
TIMEOUT_GS_TRAIN_SEC=1800
TIMEOUT_GS_METRICS_SEC=600

# 아웃라이어 필터링
OUTLIER_K_NEIGHBORS=20
OUTLIER_STD_THRESHOLD=2.0
OUTLIER_REMOVE_SMALL_CLUSTERS=true
OUTLIER_MIN_CLUSTER_RATIO=0.01

# 경로 설정
DATA_DIR=./data/jobs
GS_ROOT=./gaussian-splatting
CONDA_ENV_NAME=codyssey
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
- **3D Viewer**: PlayCanvas Model Viewer (Official)
- **Database**: SQLite (기본)
- **Async**: asyncio, aiofiles
- **Validation**: Pydantic, python-multipart

## 구현 상태

- ✅ 핵심 파이프라인 (COLMAP + Gaussian Splatting)
- ✅ 사전점검 (Preflight)
- ✅ 업로드 검증
- ✅ 단계별 진행률 추적 (step, progress)
- ✅ 평가 메트릭 (PSNR, SSIM, LPIPS)
- ✅ Train/Test Split
- ✅ 작업 대기열
- ✅ 아웃라이어 필터링
- ✅ PlayCanvas 공식 3D 뷰어 통합
- ✅ Health check (/healthz)
- ✅ GPU 메모리 모니터링
- ✅ 에러 처리 및 로깅

## 라이센스

이 프로젝트는 다음 오픈소스 프로젝트를 사용합니다:
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Inria License
- [COLMAP](https://colmap.github.io/) - BSD License
- [PlayCanvas Model Viewer](https://github.com/playcanvas/model-viewer) - MIT License

## 참고 자료

- [3D Gaussian Splatting 논문](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP 문서](https://colmap.github.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
- [PlayCanvas 문서](https://developer.playcanvas.com/)
- [Kubernetes Health Checks](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)
