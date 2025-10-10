# Gaussian Splatting 기반 3D 재구성 API 서버

본 프로젝트는 업로드된 다수의 이미지로부터 COLMAP (Structure-from-Motion)을 통해 3D 점군을 추출하고, **3D Gaussian Splatting** 기법으로 신경 방사장 모델을 학습한 뒤 결과를 웹에서 시각화하는 FastAPI 기반의 백엔드 서버입니다.

## 기능 개요

- **이미지 업로드 및 3D 재구성** – `/recon/jobs [POST]`: 사용자가 여러 장의 사진을 업로드하면 새로운 재구성 작업이 생성됩니다. 응답으로 `job_id` (작업 식별자)와 `pub_key` (공개 결과 조회 키)가 반환됩니다.

- **작업 진행 상태 조회** – `/recon/jobs/{job_id}/status [GET]`: 특정 작업의 현재 상태 (`PENDING`, `RUNNING`, `DONE`, `ERROR`), 최근 로그 내용 및 결과 뷰어 URL(완료 시)을 제공합니다.

- **3D 결과 웹 뷰어** – `/v/{pub_key} [GET]`: Three.js를 사용한 WebGL 페이지로, 해당 작업의 3D 포인트클라우드 결과를 브라우저에서 인터랙티브하게 볼 수 있습니다.

- **포인트클라우드 다운로드** – `/recon/pub/{pub_key}/cloud.ply [GET]`: 결과로 생성된 PLY 포인트클라우드 파일을 직접 다운로드하거나 뷰어에서 불러올 수 있는 엔드포인트입니다.

## 설치 및 실행

### 1. 환경 설정

Ubuntu 22.04 시스템에서, `setup.sh` 스크립트를 실행하여 의존성 라이브러리, COLMAP, Gaussian Splatting 등을 모두 설치합니다.

```bash
sudo bash setup.sh
```

**참고**:
- 이 스크립트는 자동으로 시스템의 CUDA 버전을 감지하고 적절한 PyTorch 버전을 설치합니다.
- CUDA 11.8, 12.1 등이 감지되면 해당 버전에 맞는 PyTorch가 설치됩니다.
- CUDA가 없는 경우 CPU 전용 버전이 설치됩니다.
- Python 가상환경 `gs-env`도 자동으로 생성됩니다.

설치 완료 후 다음을 통해 가상환경을 활성화하세요:

```bash
source ~/gs-env/bin/activate
```

### 2. FastAPI 의존성 설치

프로젝트 디렉토리에서 필요한 Python 패키지를 설치합니다:

```bash
cd /home/kapr/Desktop/codyssey_hackathon
pip install -r requirements.txt
```

**참고**: PyTorch는 `setup.sh` 스크립트에서 이미 설치되므로 별도로 설치할 필요가 없습니다.

### 3. 서버 실행

가상환경 활성화 후, `main.py`를 실행하여 FastAPI 서버를 시작합니다.

```bash
python main.py
```

서버가 uvicorn을 통해 `http://0.0.0.0:8000` (기본)에서 실행됩니다. Swagger UI를 통해 API 문서를 확인할 수 있습니다 (브라우저에서 `/docs` 접속).

## API 사용 예시

### 이미지 업로드 및 재구성 요청

예를 들어 `images` 폴더 내 JPG 파일들을 업로드하려면:

```bash
curl -X POST "http://localhost:8000/recon/jobs" \
  -F "files=@images/img1.jpg" \
  -F "files=@images/img2.jpg" \
  -F "files=@images/img3.jpg"
```

응답:
```json
{ "job_id": "abcd1234", "pub_key": "0123456789" }
```

### 상태 확인

```bash
curl http://localhost:8000/recon/jobs/abcd1234/status
```

응답 예:
```json
{
  "job_id": "abcd1234",
  "status": "RUNNING",
  "log_tail": [
    ">> [3/6] Sparse reconstruction (SfM) 시작...\n",
    "... (중략) ...",
    ">> [4/6] Undistorting images 및 변환된 모델 생성...\n"
  ]
}
```

작업이 `DONE` 상태가 되면 `viewer_url` 필드에 `/v/0123456789`와 같은 뷰어 URL이 포함됩니다.

### 웹 뷰어 확인

상태 응답에서 `viewer_url`을 확인하거나, 직접 브라우저에서 `http://localhost:8000/v/0123456789`로 접속하면 Three.js 기반의 3D 뷰어에서 포인트클라우드 결과를 볼 수 있습니다. 좌클릭 드래그로 회전, 우클릭 드래그로 팬, 스크롤로 줌인이 가능합니다.

## 디렉토리 구조

작업 별로 `data/jobs/{job_id}` 폴더가 생성되며, 아래와 같은 구조로 데이터가 저장됩니다:

```plaintext
data/jobs/abcd1234/
├── upload/
│   └── images/              # 업로드된 원본 이미지들
├── colmap/
│   ├── database.db          # COLMAP 데이터베이스 (특징, 매칭 저장)
│   └── sparse/0/            # COLMAP SfM 결과 (binary 모델)
├── work/
│   ├── images/              # COLMAP undistorter로 생성된 보정 이미지들
│   └── sparse/0/            # COLMAP undistorter 결과 (텍스트 파일: cameras.txt, images.txt, points3D.txt)
├── model/                   # Gaussian Splatting 모델 결과 (예: 체크포인트 등)
├── metrics/
│   └── report.json          # metrics.py에서 출력된 성능 리포트 (PSNR/SSIM 등)
├── export/
│   └── cloud.ply            # 시각화용으로 변환된 포인트 클라우드 (PLY ASCII)
├── key.txt                  # 공개 조회 키 (pub_key)을 저장
└── run.log                  # 실행 로그 (COLMAP 및 GS 출력 내용)
```

## 참고 및 추가 정보

- **실행 시간**: COLMAP과 Gaussian Splatting의 실행 과정에서 시간이 오래 걸릴 수 있습니다. 특히 Gaussian Splatting (`train.py`) 학습은 수 분에서 수 시간까지 소요될 수 있으며, 본 서버 구현에서는 기본 설정으로 전체 학습을 수행합니다.

- **동시 실행 제한**: 서버는 내부적으로 세마포어로 제어되어 동시에 최대 2개의 재구성 작업만 처리합니다. 대기 중인 작업은 이전 작업이 끝날 때까지 `PENDING` 상태로 대기합니다.

- **오류 처리**: 재구성 파이프라인 도중 오류 발생 시 해당 작업의 상태는 `ERROR`로 표시되며, `/recon/jobs/{job_id}/status` 응답의 `log_tail` 또는 `run.log` 파일을 통해 상세 원인을 확인할 수 있습니다.

- **웹 뷰어**: Three.js로 구현되어 포인트만을 렌더링하며, Gaussian Splatting으로 생성된 방사장(gaussian splats)은 포인트 클라우드로 단순화하여 보여줍니다. 보다 정교한 시각화(예: 광택 표현)는 GS의 전용 뷰어나 NerfStudio 등의 도구를 활용해야 합니다.

## 프로젝트 구조

```plaintext
codyssey_hackathon/
├── main.py              # FastAPI 서버 메인 코드
├── setup.sh             # 환경 구축 스크립트
├── requirements.txt     # Python 의존성 파일
├── README.md            # 프로젝트 문서
├── implement.pdf        # 구현 가이드 문서
└── data/                # 작업 데이터 저장 디렉토리
    └── jobs/            # 각 job_id별 작업 폴더
```

## 라이선스

본 프로젝트는 교육 및 연구 목적으로 작성되었습니다.

## 기여 및 문의

문의사항이나 개선 제안이 있으시면 이슈를 등록해주세요.
