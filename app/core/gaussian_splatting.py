"""
Gaussian Splatting training pipeline
"""
import os
import subprocess
from pathlib import Path
from app.config import settings
from app.core.pipeline import run_command
from app.utils.system import get_gpu_memory_usage
from app.utils.logger import setup_logger
from app.utils.outlier_filter import filter_outliers
from app.utils.converter import convert_ply_to_splat

logger = setup_logger(__name__)


class GaussianSplattingTrainer:
    """Gaussian Splatting training pipeline"""

    def __init__(self, work_dir: Path, output_dir: Path):
        self.work_dir = work_dir
        self.output_dir = output_dir

    async def train(self, log_file, iterations: int = None) -> Path:
        """
        Train Gaussian Splatting model

        Args:
            log_file: File handle for logging
            iterations: Number of training iterations

        Returns:
            Path to iteration directory with results
        """
        iterations = iterations or settings.TRAINING_ITERATIONS

        logger.info(f"Training Gaussian Splatting for {iterations} iterations")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # GPU memory before training
        gpu_mem_before = get_gpu_memory_usage()
        log_file.write(f">> [GPU Memory - Before Training] {gpu_mem_before}\n")
        log_file.flush()

        gs_script = settings.GAUSSIAN_SPLATTING_DIR / "train.py"

        # Setup environment
        env = os.environ.copy()
        torch_lib = settings.CONDA_PYTHON.parent.parent / "lib" / "python3.9" / "site-packages" / "torch" / "lib"
        env["LD_LIBRARY_PATH"] = f"{torch_lib}:{env.get('LD_LIBRARY_PATH', '')}"
        env["PYTHONPATH"] = str(settings.GAUSSIAN_SPLATTING_DIR)

        cmd = [
            str(settings.CONDA_PYTHON),
            str(gs_script),
            "-s", str(self.work_dir),
            "-m", str(self.output_dir),
            "--iterations", str(iterations),
            "--save_iterations", str(iterations),
            "--densify_until_iter", str(settings.DENSIFY_UNTIL_ITER),
            "--densification_interval", str(settings.DENSIFICATION_INTERVAL),
            "--opacity_reset_interval", str(settings.OPACITY_RESET_INTERVAL),
            "--resolution", "1"
        ]

        await run_command(cmd, log_file, env=env, monitor_gpu=True)

        # GPU memory after training
        gpu_mem_after = get_gpu_memory_usage()
        log_file.write(f">> [GPU Memory - After Training] {gpu_mem_after}\n")
        log_file.flush()

        iteration_dir = self.output_dir / "point_cloud" / f"iteration_{iterations}"
        return iteration_dir

    def post_process(self, iteration_dir: Path, log_file) -> None:
        """
        Post-process results: outlier filtering and format conversion

        Args:
            iteration_dir: Directory containing iteration results
            log_file: File handle for logging
        """
        ply_file = iteration_dir / "point_cloud.ply"
        filtered_ply = iteration_dir / "point_cloud_filtered.ply"
        splat_file = iteration_dir / "scene.splat"

        # Outlier filtering
        if ply_file.exists() and not filtered_ply.exists():
            log_file.write(">> Applying outlier filtering...\n")
            log_file.flush()

            try:
                filter_outliers(
                    ply_file,
                    filtered_ply,
                    k_neighbors=settings.OUTLIER_K_NEIGHBORS,
                    std_threshold=settings.OUTLIER_STD_THRESHOLD,
                    remove_small_clusters=settings.OUTLIER_REMOVE_SMALL_CLUSTERS,
                    min_cluster_ratio=settings.OUTLIER_MIN_CLUSTER_RATIO
                )
                log_file.write(">> Outlier filtering complete!\n")
            except Exception as e:
                logger.error(f"Outlier filtering failed: {e}")
                log_file.write(f">> Warning: Outlier filtering failed: {e}\n")
            log_file.flush()

        # Convert to splat
        ply_for_conversion = filtered_ply if filtered_ply.exists() else ply_file
        if ply_for_conversion.exists() and not splat_file.exists():
            log_file.write(">> Converting to splat format...\n")
            log_file.flush()

            try:
                convert_ply_to_splat(ply_for_conversion, splat_file)
                log_file.write(">> Conversion complete!\n")
            except Exception as e:
                logger.error(f"Splat conversion failed: {e}")
                log_file.write(f">> Warning: Splat conversion failed: {e}\n")
            log_file.flush()
