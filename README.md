# Gaussian Splatting 3D Reconstruction API

FastAPI 기반 3D 재구성 서비스로, 여러 이미지를 업로드하면 COLMAP과 Gaussian Splatting을 사용하여 3D 모델을 생성하고 웹 뷰어로 확인할 수 있습니다.

## 주요 기능

- **이미지 업로드**: 여러 이미지를 업로드하여 3D 재구성 작업 생성
- **COLMAP 파이프라인**: Feature extraction, matching, sparse reconstruction
- **Gaussian Splatting**: 30,000 iteration 학습 (10k/20k/30k 체크포인트)
- **점진적 업데이트**: 1만 iteration마다 결과 업데이트, 이전 체크포인트 자동 삭제
- **웹 뷰어**: Three.js 기반 인터랙티브 3D 뷰어
- **동시 작업**: 최대 2개 작업 동시 실행

## 설치 방법

### 1. 환경 설정

```bash
# Conda 환경 생성 (Python 3.9)
conda create -n codyssey python=3.9
conda activate codyssey

# PyTorch 설치
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# COLMAP 설치 (Ubuntu)
sudo apt-get install colmap
```

### 2. Gaussian Splatting 설정

```bash
# Gaussian Splatting 클론
git clone https://github.com/graphdeco-inria/gaussian-splatting --recursive
cd gaussian-splatting

# 서브모듈 빌드
cd submodules/diff-gaussian-rasterization
python setup.py build_ext --inplace

cd ../simple-knn
python setup.py build_ext --inplace

cd ../fused-ssim
python setup.py build_ext --inplace

cd ../../..
```

### 3. 프로젝트 설정

```bash
# 의존성 설치
pip install -r requirements.txt

# NumPy 버전 확인 (2.x인 경우 다운그레이드)
pip install "numpy<2"
```

## 실행 방법

```bash
# 서버 실행
python main.py

# 또는 uvicorn 직접 실행
uvicorn main:app --host 0.0.0.0 --port 8000
```

서버가 시작되면 http://localhost:8000 에서 접근 가능합니다.

## API 사용법

### 1. 작업 생성 (이미지 업로드)

```bash
curl -X POST http://localhost:8000/recon/jobs \
  -F "files=@image1.jpg" \
  -F "files=@image2.jpg" \
  -F "files=@image3.jpg"
```

**응답 예시:**
```json
{
  "job_id": "abc12345",
  "pub_key": "1234567890"
}
```

### 2. 작업 상태 확인

```bash
curl http://localhost:8000/recon/jobs/{job_id}/status
```

**응답 예시:**
```json
{
  "job_id": "abc12345",
  "status": "RUNNING",
  "log_tail": ["...", "..."],
  "viewer_url": "/v/1234567890"
}
```

**상태 값:**
- `PENDING`: 대기 중
- `RUNNING`: 실행 중
- `DONE`: 완료
- `ERROR`: 오류 발생

### 3. 결과 확인

#### 웹 뷰어
```
http://localhost:8000/v/{pub_key}
```

#### PLY 파일 다운로드
```bash
curl -O http://localhost:8000/recon/pub/{pub_key}/cloud.ply
```

#### Splat 파일 다운로드
```bash
curl -O http://localhost:8000/recon/pub/{pub_key}/scene.splat
```

## 학습 진행 과정

1. **COLMAP 단계** (약 1-5분)
   - Feature extraction
   - Feature matching
   - Sparse reconstruction
   - Image undistortion

2. **Gaussian Splatting 학습** (약 30-60분)
   - **10,000 iterations**: 첫 번째 체크포인트 생성
   - **20,000 iterations**: 두 번째 체크포인트 생성 (10k 삭제)
   - **30,000 iterations**: 최종 결과 생성 (20k 삭제)

각 체크포인트마다 PLY → splat 자동 변환이 수행됩니다.

## 뷰어 사용법

### 마우스 컨트롤
- **왼쪽 클릭 + 드래그**: 회전
- **오른쪽 클릭 + 드래그**: 팬(이동)
- **스크롤**: 줌 인/아웃

### 뷰어 커스터마이징

점 크기를 조정하려면 `viewer_template.html` 파일의 170번째 줄을 수정하세요:

```javascript
const material = new THREE.PointsMaterial({
    size: 0.025,        // 이 값을 조정 (0.01 ~ 0.5)
    vertexColors: true,
    sizeAttenuation: true
});
```

## 프로젝트 구조

```
codyssey_hackathon/
├── main.py                 # FastAPI 서버 메인 파일
├── convert_to_splat.py     # PLY → splat 변환 유틸리티
├── viewer_template.html    # Three.js 웹 뷰어
├── requirements.txt        # Python 의존성
├── .gitignore             # Git 제외 파일 목록
├── README.md              # 이 파일
├── gaussian-splatting/    # Gaussian Splatting 레포지토리
├── data/                  # 작업 데이터 저장 폴더
│   └── jobs/
│       └── {job_id}/
│           ├── upload/    # 업로드된 이미지
│           ├── colmap/    # COLMAP 출력
│           ├── work/      # 처리된 데이터
│           ├── gs_output/ # Gaussian Splatting 출력
│           │   └── point_cloud/
│           │       └── iteration_{10k,20k,30k}/
│           │           ├── point_cloud.ply
│           │           └── scene.splat
│           └── run.log    # 작업 로그
└── static/                # 정적 파일 (antimatter15 viewer 등)
```

## 체크포인트 관리

### 자동 정리
- 각 체크포인트 완료 시 이전 체크포인트 자동 삭제
- 디스크 공간 효율적 관리

### 체크포인트 우선순위
API는 다음 순서로 최신 체크포인트를 찾습니다:
1. `iteration_30000/` (최종)
2. `iteration_20000/` (중간)
3. `iteration_10000/` (초기)
4. `iteration_7000/` (레거시)

## 문제 해결

### GPU 메모리 부족
```bash
# GPU 메모리 확인
nvidia-smi

# iteration 수 줄이기 (main.py:231)
"--iterations", "10000"  # 30000 → 10000
```

### COLMAP 오류
```bash
# COLMAP 버전 확인
colmap -h

# 이미지 품질 확인 (흐릿하거나 유사한 이미지는 제외)
```

### NumPy 버전 오류
```bash
# NumPy 다운그레이드
pip install "numpy<2"
```

## 서버 재시작 후 복구

서버가 재시작되면 메모리의 job 정보가 사라집니다. 임시 복구 방법:

```bash
curl -X POST http://localhost:8000/admin/restore-job \
  -H "Content-Type: application/json" \
  -d '{"job_id": "abc12345", "pub_key": "1234567890", "status": "DONE"}'
```

## 기술 스택

- **FastAPI**: 웹 API 프레임워크
- **COLMAP**: Structure-from-Motion 재구성
- **Gaussian Splatting**: 3D 장면 표현
- **Three.js**: WebGL 기반 3D 렌더링
- **asyncio**: 비동기 작업 처리

## 라이센스

이 프로젝트는 다음 오픈소스 프로젝트를 사용합니다:
- [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) - Inria license
- [COLMAP](https://colmap.github.io/) - BSD license
- [Three.js](https://threejs.org/) - MIT license

## 참고 자료

- [Gaussian Splatting 논문](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [COLMAP 문서](https://colmap.github.io/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)
