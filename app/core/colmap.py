"""
COLMAP pipeline for Structure-from-Motion reconstruction
"""
import shutil
from pathlib import Path
from app.config import settings
from app.core.pipeline import run_command
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


class COLMAPPipeline:
    """COLMAP pipeline for Structure-from-Motion reconstruction"""

    def __init__(self, job_dir: Path):
        self.job_dir = job_dir
        self.database_path = job_dir / "colmap" / "database.db"
        self.images_path = job_dir / "upload" / "images"
        self.sparse_path = job_dir / "colmap" / "sparse"
        self.work_path = job_dir / "work"

    async def extract_features(self, log_file) -> None:
        """Extract SIFT features from images"""
        logger.info(f"Extracting features for job {self.job_dir.name}")

        cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_path),
            "--ImageReader.camera_model", settings.COLMAP_CAMERA_MODEL,
            "--SiftExtraction.max_num_features", str(settings.COLMAP_MAX_FEATURES),
            "--FeatureExtraction.num_threads", str(settings.COLMAP_NUM_THREADS)
        ]

        await run_command(cmd, log_file)

    async def match_features(self, log_file) -> None:
        """Match features between images"""
        logger.info(f"Matching features for job {self.job_dir.name}")

        cmd = [
            "colmap", "exhaustive_matcher",
            "--database_path", str(self.database_path),
            "--FeatureMatching.num_threads", str(settings.COLMAP_NUM_THREADS)
        ]

        await run_command(cmd, log_file)

    async def reconstruct(self, log_file) -> Path:
        """Perform sparse reconstruction (SfM)"""
        logger.info(f"Reconstructing sparse model for job {self.job_dir.name}")

        self.sparse_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "colmap", "mapper",
            "--database_path", str(self.database_path),
            "--image_path", str(self.images_path),
            "--output_path", str(self.sparse_path)
        ]

        await run_command(cmd, log_file)

        model0_path = self.sparse_path / "0"
        if not model0_path.exists():
            raise RuntimeError("COLMAP reconstruction failed: no sparse model generated")

        return model0_path

    async def undistort_images(self, model_path: Path, log_file) -> Path:
        """Undistort images and prepare for Gaussian Splatting"""
        logger.info(f"Undistorting images for job {self.job_dir.name}")

        self.work_path.mkdir(parents=True, exist_ok=True)

        cmd = [
            "colmap", "image_undistorter",
            "--image_path", str(self.images_path),
            "--input_path", str(model_path),
            "--output_path", str(self.work_path),
            "--output_type", "COLMAP"
        ]

        await run_command(cmd, log_file)

        # GS expects sparse model in sparse/0/ subdirectory
        sparse_0_dir = self.work_path / "sparse" / "0"
        sparse_0_dir.mkdir(parents=True, exist_ok=True)

        # Move sparse files
        sparse_dir = self.work_path / "sparse"
        for item in sparse_dir.iterdir():
            if item.is_file():
                shutil.move(str(item), str(sparse_0_dir / item.name))

        return self.work_path

    async def convert_to_text(self, sparse_dir: Path, log_file) -> None:
        """Convert COLMAP binary model to text format"""
        logger.info(f"Converting COLMAP model to text format")

        cmd = [
            "colmap", "model_converter",
            "--input_path", str(sparse_dir),
            "--output_path", str(sparse_dir),
            "--output_type", "TXT"
        ]

        await run_command(cmd, log_file)

    def create_train_test_split(self, work_dir: Path, test_ratio: float = 0.2) -> None:
        """Create train/test split for evaluation"""
        import random

        images_dir = work_dir / "images"
        if not images_dir.exists():
            logger.warning(f"Images directory not found: {images_dir}")
            return

        # Get all image files
        image_files = sorted(list(images_dir.glob("*.*")))
        if len(image_files) < 3:
            logger.warning(f"Not enough images for train/test split: {len(image_files)}")
            return

        # Calculate test set size (at least 1 image, at most test_ratio of total)
        test_count = max(1, int(len(image_files) * test_ratio))

        # Randomly select test images
        random.seed(42)  # For reproducibility
        test_indices = set(random.sample(range(len(image_files)), test_count))

        # Create train.txt and test.txt
        train_file = work_dir / "train.txt"
        test_file = work_dir / "test.txt"

        with open(train_file, 'w') as train_f, open(test_file, 'w') as test_f:
            for idx, img_file in enumerate(image_files):
                img_name = img_file.name
                if idx in test_indices:
                    test_f.write(f"{img_name}\n")
                else:
                    train_f.write(f"{img_name}\n")

        logger.info(f"Created train/test split: {len(image_files) - test_count} train, {test_count} test")
