import os
import uuid
import random
import asyncio
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse, HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# FastAPI 앱 생성
app = FastAPI(
    title="Gaussian Splatting 3D Reconstruction API",
    description="Upload images to reconstruct a 3D model using Gaussian Splatting and view the result in a web viewer."
)

# CORS 등 필요한 미들웨어 설정 (필요에 따라)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Mount static files directory (if it exists)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# 작업 관리용 전역 변수
jobs = {}  # job_id -> 정보(dict: status, pub_key, etc.)
pub_to_job = {}  # pub_key -> job_id 매핑
MAX_CONCURRENT = 2
sem = asyncio.Semaphore(MAX_CONCURRENT)  # 동시 실행 세마포어 (최대 2개 작업)

# 경로 기본 설정 - 현재 디렉토리 내에 data/jobs 폴더 사용
BASE_DIR = Path(__file__).resolve().parent / "data" / "jobs"
BASE_DIR.mkdir(parents=True, exist_ok=True)


# COLMAP 및 Gaussian Splatting 실행을 비동기 처리하기 위한 유틸리티 함수
async def run_command(cmd: list, log_file, cwd: Path = None, env: dict = None):
    """
    주어진 명령(cmd 리스트)을 서브프로세스로 실행하고, 출력 스트림을 실시간 로그 파일에 기록합니다.
    오류 발생 시 예외를 발생시켜 상위에서 처리합니다.
    """
    # 서브프로세스 실행 (표준 출력과 에러를 합쳐 한 스트림으로 처리)
    process = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=(str(cwd) if cwd else None),
        env=env,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT
    )

    # 출력 스트림을 읽어가며 로그 파일에 기록
    # tqdm progress bar는 매우 긴 한 줄로 나올 수 있으므로 chunk 단위로 읽음
    while True:
        try:
            # 작은 청크로 읽어서 타임아웃 방지
            chunk = await asyncio.wait_for(process.stdout.read(4096), timeout=1.0)
            if not chunk:
                break
            text = chunk.decode(errors="ignore")
            log_file.write(text)
            log_file.flush()
        except asyncio.TimeoutError:
            # 타임아웃 발생 시 프로세스가 아직 실행 중인지 확인
            if process.returncode is not None:
                break
            # 아직 실행 중이면 계속 읽기
            continue

    # 프로세스 종료 대기
    exit_code = await process.wait()
    if exit_code != 0:
        # 에러가 발생한 경우, 로그 파일에 오류 메시지 기록 후 예외 발생
        log_file.write(f"[ERROR] Command {' '.join(cmd)} exited with code {exit_code}\n")
        log_file.flush()
        raise RuntimeError(f"Command failed: {' '.join(cmd)} (exit code: {exit_code})")


# COLMAP points3D.txt -> PLY 변환 함수
def convert_points3d_to_ply(points3d_txt: Path, ply_path: Path):
    """
    COLMAP에서 생성된 points3D.txt 파일을 읽어 PLY (ASCII 포인트클라우드) 파일로 저장합니다.
    """
    with open(points3d_txt, 'r') as f_txt, open(ply_path, 'w') as f_ply:
        lines = f_txt.readlines()

        # points3D.txt의 유효 데이터 라인만 파싱 (주석 줄 제외)
        points = []
        for line in lines:
            if line.strip().startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            # points3D.txt 포맷: POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]...
            if len(parts) >= 7:
                X, Y, Z = map(float, parts[1:4])
                R, G, B = map(int, parts[4:7])
                points.append((X, Y, Z, R, G, B))

        # PLY 헤더 작성
        f_ply.write("ply\nformat ascii 1.0\n")
        f_ply.write(f"element vertex {len(points)}\n")
        f_ply.write("property float x\nproperty float y\nproperty float z\n")
        f_ply.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f_ply.write("end_header\n")

        # 각 포인트 좌표와 색상 기록
        for (x, y, z, r, g, b) in points:
            f_ply.write(f"{x:.6f} {y:.6f} {z:.6f} {r} {g} {b}\n")


# 백그라운드에서 실행될 3D 재구성 파이프라인 함수
async def process_job(job_id: str):
    """
    주어진 job_id에 대해 COLMAP 및 Gaussian Splatting 파이프라인을 실행합니다.
    """
    # 세마포어 획득: 동시에 최대 2개 작업만 실행
    async with sem:
        job = jobs[job_id]
        job_dir = Path(BASE_DIR / job_id)
        log_path = job_dir / "run.log"

        # 로그 파일 열기 (추가 모드)
        with open(log_path, 'a', encoding='utf-8', buffering=1) as log_file:
            try:
                # 상태 갱신: 실행 중
                job["status"] = "RUNNING"
                log_file.write(f"=== Job {job_id} started (pub_key={job['pub_key']}) ===\n")
                log_file.flush()

                # COLMAP feature extraction 단계
                log_file.write(">> [1/6] Feature extraction 시작...\n")
                log_file.flush()

                images_path = job_dir / "upload" / "images"
                database_path = job_dir / "colmap" / "database.db"

                # colmap 실행: feature_extractor
                await run_command([
                    "colmap", "feature_extractor",
                    "--database_path", str(database_path),
                    "--image_path", str(images_path),
                    "--ImageReader.camera_model", "OPENCV",  # 기본 카메라 모델 (OPENCV 핀홀)
                    "--FeatureExtraction.num_threads", "8"
                ], log_file)

                # colmap 실행: exhaustive_matcher (전역 매칭)
                log_file.write(">> [2/6] Feature matching 시작...\n")
                log_file.flush()
                await run_command([
                    "colmap", "exhaustive_matcher",
                    "--database_path", str(database_path),
                    "--FeatureMatching.num_threads", "8"
                ], log_file)

                # colmap 실행: mapper (Structure-from-Motion 재구성)
                log_file.write(">> [3/6] Sparse reconstruction (SfM) 시작...\n")
                log_file.flush()

                sparse_path = job_dir / "colmap" / "sparse"
                sparse_path.mkdir(parents=True, exist_ok=True)  # 출력 디렉토리 생성

                await run_command([
                    "colmap", "mapper",
                    "--database_path", str(database_path),
                    "--image_path", str(images_path),
                    "--output_path", str(sparse_path)
                ], log_file)

                # sparse 결과 (0번 모델) 존재 여부 검사
                model0_path = sparse_path / "0"
                if not model0_path.exists():
                    # 재구성된 모델이 없으면 오류로 처리
                    raise RuntimeError("COLMAP reconstruction failed: no sparse model generated.")

                # colmap 실행: image_undistorter (이미지 왜곡 보정 및 모델 변환)
                log_file.write(">> [4/6] Undistorting images 및 변환된 모델 생성...\n")
                log_file.flush()

                work_path = job_dir / "work"
                work_path.mkdir(parents=True, exist_ok=True)
                undistort_sparse = work_path / "sparse"
                undistort_images = work_path / "images"

                await run_command([
                    "colmap", "image_undistorter",
                    "--image_path", str(images_path),
                    "--input_path", str(model0_path),
                    "--output_path", str(work_path),
                    "--output_type", "COLMAP"
                ], log_file)

                # GS expects sparse model in sparse/0/ subdirectory
                sparse_0_dir = work_path / "sparse" / "0"
                sparse_0_dir.mkdir(parents=True, exist_ok=True)

                # Move sparse files to sparse/0/ subdirectory
                sparse_dir = work_path / "sparse"
                for item in sparse_dir.iterdir():
                    if item.is_file():
                        import shutil
                        shutil.move(str(item), str(sparse_0_dir / item.name))

                # Gaussian Splatting 학습 시작
                log_file.write(">> [5/6] Gaussian Splatting 학습 (train.py) 시작...\n")
                log_file.flush()

                gs_output_dir = job_dir / "gs_output"
                gs_output_dir.mkdir(parents=True, exist_ok=True)

                # conda 환경에서 Gaussian Splatting 실행 - 직접 python 실행
                import os
                conda_python = os.path.expanduser("~/miniconda3/envs/codyssey/bin/python")
                gs_script = str(Path(__file__).resolve().parent / "gaussian-splatting" / "train.py")

                # 환경 변수 설정
                env = os.environ.copy()
                torch_lib = os.path.expanduser("~/miniconda3/envs/codyssey/lib/python3.9/site-packages/torch/lib")
                env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
                env["PYTHONPATH"] = str(Path(__file__).resolve().parent / "gaussian-splatting")

                # Train for 30000 iterations with checkpoints at 10k, 20k, 30k
                gs_train_cmd = [
                    conda_python,
                    gs_script,
                    "-s", str(work_path),
                    "-m", str(gs_output_dir),
                    "--iterations", "30000",
                    "--save_iterations", "10000", "20000", "30000"
                ]

                try:
                    await run_command(gs_train_cmd, log_file, env=env)
                    log_file.write(">> [6/6] Gaussian Splatting 학습 완료!\n")
                    log_file.flush()

                    # Convert and cleanup: keep only the latest checkpoint
                    import subprocess
                    point_cloud_dir = gs_output_dir / "point_cloud"

                    # Process each checkpoint
                    for iteration in [10000, 20000, 30000]:
                        iter_dir = point_cloud_dir / f"iteration_{iteration}"
                        if iter_dir.exists():
                            ply_file = iter_dir / "point_cloud.ply"
                            splat_file = iter_dir / "scene.splat"

                            # Convert PLY to splat format
                            if ply_file.exists() and not splat_file.exists():
                                log_file.write(f">> Converting iteration {iteration} to splat format...\n")
                                log_file.flush()
                                convert_cmd = [
                                    "python",
                                    str(Path(__file__).resolve().parent / "convert_to_splat.py"),
                                    str(ply_file),
                                    str(splat_file)
                                ]
                                subprocess.run(convert_cmd, check=True)

                            # Delete previous checkpoint
                            if iteration > 10000:
                                prev_iteration = iteration - 10000
                                prev_dir = point_cloud_dir / f"iteration_{prev_iteration}"
                                if prev_dir.exists():
                                    import shutil
                                    log_file.write(f">> Removing previous checkpoint: iteration_{prev_iteration}\n")
                                    log_file.flush()
                                    shutil.rmtree(prev_dir)

                except Exception as gs_error:
                    log_file.write(f"[WARNING] Gaussian Splatting failed: {str(gs_error)}\n")
                    log_file.write("Continuing with COLMAP results only...\n")
                    log_file.flush()

                # point cloud (.ply) 생성 - COLMAP sparse 점들을 PLY 포맷으로 내보내기
                export_dir = job_dir / "export"
                export_dir.mkdir(parents=True, exist_ok=True)
                ply_path = export_dir / "cloud.ply"

                # sparse 폴더에서 points3D 파일 찾기 (bin 또는 txt)
                # Files were moved to sparse/0/ subdirectory for GS compatibility
                sparse_dir = work_path / "sparse" / "0"
                points3d_bin = sparse_dir / "points3D.bin"
                points3d_txt = sparse_dir / "points3D.txt"

                # 바이너리 파일이 있으면 텍스트로 변환
                if points3d_bin.exists() and not points3d_txt.exists():
                    log_file.write(">> Converting COLMAP binary to text format...\n")
                    log_file.flush()
                    await run_command([
                        "colmap", "model_converter",
                        "--input_path", str(sparse_dir),
                        "--output_path", str(sparse_dir),
                        "--output_type", "TXT"
                    ], log_file)

                if not points3d_txt.exists():
                    raise RuntimeError("points3D.txt not found, cannot generate cloud.ply")

                convert_points3d_to_ply(points3d_txt, ply_path)

                # 공개용 키 저장
                with open(job_dir / "key.txt", 'w') as f:
                    f.write(job["pub_key"])

                # 상태 갱신: 완료
                job["status"] = "DONE"
                log_file.write(f"=== Job {job_id} 완료: 성공적으로 처리되었습니다. ===\n")
                log_file.flush()

            except Exception as e:
                # 오류 발생: 상태 표시 및 로그 기록
                job["status"] = "ERROR"
                error_msg = f"[Exception] {str(e)}"
                log_file.write(error_msg + "\n")
                log_file.write(f"=== Job {job_id} 종료: 오류 발생 ===\n")
                log_file.flush()
                print(error_msg)  # 서버 콘솔에도 출력

        # (세마포어는 async with 블록에서 자동 해제됨)


# API 엔드포인트 구현
@app.post("/recon/jobs", summary="새 3D 재구성 작업 생성", description="여러 이미지를 업로드하여 3D 재구성 작업을 생성합니다. 작업 ID와 공개 뷰어 키를 반환합니다.")
async def create_reconstruction_job(files: list[UploadFile] = File(...)):
    # 새로운 job 식별자 생성 (UUID4 기반 8문자 또는 timestamp 등)
    job_id = uuid.uuid4().hex[:8]  # 8자리 16진수 ID

    # 10자리 공개용 랜덤 키 생성 (숫자)
    pub_key = ''.join(random.choice("0123456789") for _ in range(10))

    # 혹시 pub_key 충돌 시 변경
    while pub_key in pub_to_job:
        pub_key = ''.join(random.choice("0123456789") for _ in range(10))

    # 작업 디렉터리들 생성
    job_dir = Path(BASE_DIR / job_id)
    (job_dir / "upload" / "images").mkdir(parents=True, exist_ok=True)
    (job_dir / "colmap").mkdir(parents=True, exist_ok=True)

    # 업로드된 이미지들을 저장
    for file in files:
        # 파일 이름이 중복되면 덮어쓰지 않도록 고유 이름 추가 가능 (여기서는 그대로 사용)
        content = await file.read()
        img_path = job_dir / "upload" / "images" / file.filename
        with open(img_path, 'wb') as f:
            f.write(content)

    # 작업 초기 상태 저장
    jobs[job_id] = {
        "status": "PENDING",  # 아직 대기 상태
        "pub_key": pub_key
    }
    pub_to_job[pub_key] = job_id

    # 백그라운드로 파이프라인 작업 수행 시작
    asyncio.create_task(process_job(job_id))

    # 작업 ID와 공개 키 반환
    return {"job_id": job_id, "pub_key": pub_key}


@app.get("/recon/jobs/{job_id}/status", summary="작업 상태 조회", description="지정한 job_id에 대한 현재 상태와 로그 일부, 결과 확인 URL 등을 반환합니다.")
async def get_job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job_info = jobs[job_id]
    status = job_info.get("status", "PENDING")

    # run.log에서 마지막 N줄 읽기
    log_path = Path(BASE_DIR / job_id / "run.log")
    log_tail = []
    if log_path.exists():
        try:
            with open(log_path, 'r') as f:
                lines = f.readlines()
                # 최근 10줄만 추출
                if len(lines) > 0:
                    log_tail = lines[-10:]
        except Exception as e:
            log_tail = [f"(log read error: {e})"]

    response = {
        "job_id": job_id,
        "status": status,
        "log_tail": log_tail
    }

    # 완료된 경우 결과 뷰어 URL 포함
    if status == "DONE":
        response["viewer_url"] = f"/v/{job_info['pub_key']}"
    elif status == "ERROR":
        # 오류인 경우 메시지를 로그 tail에 담고, 필요하면 에러 설명 추가
        response["error"] = "An error occurred. Check log for details."

    return JSONResponse(content=response)


@app.get("/v/{pub_key}", summary="3D 결과 뷰어 페이지", response_class=HTMLResponse)
async def view_result_page(pub_key: str, request: Request):
    # 주어진 공개 키에 해당하는 job 찾기
    job_id = pub_to_job.get(pub_key)
    if job_id is None or job_id not in jobs:
        raise HTTPException(status_code=404, detail="Invalid viewer key")

    # 작업이 완료되었는지 확인
    if jobs[job_id].get("status") != "DONE":
        # 아직 완료되지 않은 경우 대기 메시지 표시 (간단한 HTML 반환)
        return HTMLResponse(content="<html><body><h3>Result is not ready yet. Please check again later.</h3></body></html>")

    # Load simple Three.js viewer template
    viewer_path = Path(__file__).parent / "viewer_template.html"
    with open(viewer_path, 'r') as f:
        html_content = f.read()

    # Use relative URL for same-origin request
    splat_url = f"/recon/pub/{pub_key}/scene.splat"
    html_content = html_content.replace('SPLAT_URL_PLACEHOLDER', splat_url)
    return HTMLResponse(content=html_content, status_code=200)

@app.get("/recon/pub/{pub_key}/cloud.ply", summary="생성된 포인트클라우드 PLY 파일 다운로드")
async def download_ply(pub_key: str):
    # 공개 키로부터 job 검색
    job_id = pub_to_job.get(pub_key)
    if job_id is None or job_id not in jobs:
        raise HTTPException(status_code=404, detail="Invalid key or job not found")

    if jobs[job_id].get("status") != "DONE":
        raise HTTPException(status_code=400, detail="Result not ready")

    # Gaussian Splatting 결과를 먼저 확인 (더 좋은 품질) - latest checkpoint
    point_cloud_dir = Path(BASE_DIR / job_id / "gs_output" / "point_cloud")
    for iteration in [30000, 20000, 10000, 7000]:
        gs_ply = point_cloud_dir / f"iteration_{iteration}" / "point_cloud.ply"
        if gs_ply.exists():
            return FileResponse(path=gs_ply, filename="cloud.ply", media_type="application/octet-stream")

    # GS 결과가 없으면 COLMAP sparse 결과 사용
    colmap_ply = Path(BASE_DIR / job_id / "export" / "cloud.ply")
    if colmap_ply.exists():
        return FileResponse(path=colmap_ply, filename="cloud.ply", media_type="application/octet-stream")

    raise HTTPException(status_code=404, detail="cloud.ply not found")


@app.get("/recon/pub/{pub_key}/scene.splat", summary="Gaussian Splatting .splat 파일 다운로드")
async def download_splat(pub_key: str):
    # 공개 키로부터 job 검색
    job_id = pub_to_job.get(pub_key)
    if job_id is None or job_id not in jobs:
        raise HTTPException(status_code=404, detail="Invalid key or job not found")

    if jobs[job_id].get("status") != "DONE":
        raise HTTPException(status_code=400, detail="Result not ready")

    # Find the latest checkpoint (check 30k, 20k, 10k in order)
    point_cloud_dir = Path(BASE_DIR / job_id / "gs_output" / "point_cloud")
    for iteration in [30000, 20000, 10000, 7000]:
        splat_file = point_cloud_dir / f"iteration_{iteration}" / "scene.splat"
        if splat_file.exists():
            # Add iteration info as custom header
            from fastapi import Response
            with open(splat_file, 'rb') as f:
                content = f.read()
            return Response(
                content=content,
                media_type="application/octet-stream",
                headers={
                    "X-Iteration": str(iteration),
                    "Content-Disposition": f"attachment; filename=scene_{iteration}.splat"
                }
            )

    raise HTTPException(status_code=404, detail="scene.splat not found")


@app.post("/admin/restore-job", summary="서버 재시작 후 job 정보 복구 (임시)")
async def restore_job(data: dict):
    """서버가 재시작되어 메모리가 초기화된 경우 job 정보를 복구합니다."""
    job_id = data.get("job_id")
    pub_key = data.get("pub_key")
    status = data.get("status", "DONE")

    jobs[job_id] = {
        "status": status,
        "pub_key": pub_key
    }
    pub_to_job[pub_key] = job_id
    return {"message": f"Job {job_id} restored successfully", "pub_key": pub_key}


# (선택) 애플리케이션 실행 설정
if __name__ == "__main__":
    import uvicorn
    # uvicorn 서버 실행: 필요에 따라 host/port 조정
    uvicorn.run(app, host="0.0.0.0", port=8000)
