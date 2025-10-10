#!/bin/bash
# Ubuntu 22.04 + CUDA 환경에서 COLMAP 및 Gaussian Splatting 설치 스크립트
# 이 스크립트를 root 권한으로 실행하거나, sudo를 사용하여 필요한 부분을 설치합니다.

echo "=== [1/6] 시스템 패키지 업데이트 및 필수 패키지 설치 ==="
sudo apt-get update

# COLMAP 빌드에 필요한 개발 라이브러리 설치 (Qt5 GUI 비활성화 가능하지만, 여기서는 포함하여 설치)
sudo apt-get install -y git build-essential cmake ninja-build \
    libboost-program-options-dev libboost-graph-dev libboost-system-dev \
    libeigen3-dev libfreeimage-dev libmetis-dev libgoogle-glog-dev \
    libgtest-dev libgmock-dev libsqlite3-dev libglew-dev \
    qtbase5-dev libqt5opengl5-dev libcgal-dev libceres-dev \
    libcurl4-openssl-dev

# (참고) Ubuntu 22.04의 기본 CUDA 툴킷(GPU 가속)과 GCC 호환 문제 해결:
echo "=== [2/6] GCC 컴파일러 설정 (필요 시 gcc-10 사용) ==="
sudo apt-get install -y gcc-10 g++-10
export CC=/usr/bin/gcc-10
export CXX=/usr/bin/g++-10
export CUDAHOSTCXX=/usr/bin/g++-10

# CUDA 경로 확인 및 설정
echo "=== [2.5/6] CUDA 환경 확인 ==="
if [ -d "/usr/local/cuda" ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export CUDA_HOME=/usr/local/cuda
    echo "CUDA found at: $(which nvcc)"
    echo "CUDA version: $(nvcc --version | grep release)"

    # GPU 아키텍처 감지 (nvidia-smi 사용)
    if command -v nvidia-smi &> /dev/null; then
        COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | tr -d '.')
        echo "Detected GPU compute capability: ${COMPUTE_CAP}"
        CUDA_ENABLED="-DCMAKE_CUDA_ARCHITECTURES=${COMPUTE_CAP}"
    else
        echo "Warning: nvidia-smi not found. Using common CUDA architectures."
        # 일반적인 아키텍처들 (RTX 30xx/40xx, V100, A100 등)
        CUDA_ENABLED="-DCMAKE_CUDA_ARCHITECTURES=75;80;86;89;90"
    fi
else
    echo "Warning: CUDA not found. Building COLMAP without CUDA support."
    CUDA_ENABLED="-DCUDA_ENABLED=OFF"
fi

echo "=== [3/6] COLMAP 소스 다운로드 및 빌드 ==="
# COLMAP 저장소 클론 (v3.8 안정 버전 체크아웃 가능; 여기서는 최신 버전 사용)
if [ ! -d ~/colmap ]; then
    git clone https://github.com/colmap/colmap.git ~/colmap
fi
cd ~/colmap
git submodule update --init --recursive  # 필요한 서브모듈 (glog, gtest 등) 초기화

rm -rf build
mkdir build && cd build

# CMake 구성: GUI 비활성화 및 CUDA 아키텍처 설정 (있는 경우)
cmake .. -GNinja ${CUDA_ENABLED} -DBUILD_GUI=OFF

# 컴파일 및 설치
ninja -j$(nproc)
sudo ninja install  # colmap 명령을 /usr/local/bin 에 설치

# COLMAP 명령어 확인
if ! command -v colmap &> /dev/null; then
    echo "COLMAP 설치 실패: 'colmap' 명령을 찾을 수 없습니다."
    exit 1
fi

echo "COLMAP 설치 완료. 버전: $(colmap --version)"

echo "=== [4/6] Python 가상환경 및 Gaussian Splatting 설치 ==="
# Python3 venv 생성 및 활성화
if [ ! -d ~/gs-env ]; then
    python3 -m venv ~/gs-env
fi
source ~/gs-env/bin/activate

# Python 패키지 업그레이드
pip install --upgrade pip

# PyTorch 설치 (CUDA 버전에 맞춰서)
echo "=== [5/6] PyTorch 설치 ==="
if [ -d "/usr/local/cuda" ]; then
    # CUDA 버전 확인
    CUDA_VERSION=$(nvcc --version | grep release | sed -n 's/.*release \([0-9]*\)\.\([0-9]*\).*/\1\2/p')
    echo "Detected CUDA version: ${CUDA_VERSION}"

    if [ "${CUDA_VERSION}" == "118" ]; then
        echo "Installing PyTorch for CUDA 11.8..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [ "${CUDA_VERSION}" == "121" ]; then
        echo "Installing PyTorch for CUDA 12.1..."
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        echo "Installing PyTorch with default CUDA support..."
        pip install torch torchvision torchaudio
    fi
else
    echo "Installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Gaussian Splatting 레포지토리 클론 (서브모듈 포함)
if [ ! -d ~/gaussian-splatting ]; then
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git ~/gaussian-splatting
fi
cd ~/gaussian-splatting

# 요구 Python 패키지 설치
pip install -r requirements.txt

# (주의) pycolmap 설치 문제 시 특정 버전으로 재설치
pip install pycolmap==0.4.0

# Gaussian Splatting CUDA 확장 빌드 (CUDA가 있는 경우에만)
if [ -d "/usr/local/cuda" ]; then
    echo "Building Gaussian Splatting CUDA extensions..."
    python setup.py build_ext --inplace
else
    echo "Skipping CUDA extension build (CUDA not available)"
fi

echo "=== [6/6] 환경 구성 완료 ==="
echo "가상환경 'gs-env'이 생성되었으며, COLMAP 및 Gaussian Splatting이 설치되었습니다."
echo "FastAPI 서버 코드(main.py) 실행 전에 'source ~/gs-env/bin/activate'로 가상환경을 활성화하세요."
echo "또한, data/jobs 디렉토리가 존재하고 애플리케이션이 해당 경로에 쓸 수 있는지 확인하십시오."
echo ""
echo "현재 설치된 환경:"
echo "  - COLMAP: $(which colmap 2>/dev/null || echo 'Not installed')"
echo "  - Python: $(python3 --version)"
echo "  - CUDA: $([ -d '/usr/local/cuda' ] && nvcc --version | grep release || echo 'Not available')"
