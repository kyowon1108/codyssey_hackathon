"""
Application configuration settings
"""
import os
from pathlib import Path
from typing import Optional


class Settings:
    """Application settings and configuration"""

    # Project paths
    BASE_DIR: Path = Path(__file__).parent.parent
    DATA_DIR: Path = BASE_DIR / "data" / "jobs"
    GAUSSIAN_SPLATTING_DIR: Path = BASE_DIR / "gaussian-splatting"
    TEMPLATES_DIR: Path = BASE_DIR / "templates"
    STATIC_DIR: Path = BASE_DIR / "static"
    LOGS_DIR: Path = BASE_DIR / "logs"

    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./gaussian_splatting.db")

    # API Settings
    APP_TITLE: str = "Gaussian Splatting 3D Reconstruction API"
    APP_DESCRIPTION: str = "Upload images to reconstruct a 3D model using Gaussian Splatting"
    APP_VERSION: str = "2.0.0"
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    BASE_URL: str = os.getenv("BASE_URL", "http://kaprpc.iptime.org:5051")
    DEBUG: bool = os.getenv("DEBUG", "False").lower() == "true"

    # CORS
    CORS_ORIGINS: list = ["*"]

    # Processing limits
    MAX_CONCURRENT_JOBS: int = int(os.getenv("MAX_CONCURRENT_JOBS", "2"))
    MIN_IMAGES: int = int(os.getenv("MIN_IMAGES", "3"))
    MAX_IMAGES: int = int(os.getenv("MAX_IMAGES", "200"))
    MAX_IMAGE_SIZE: int = int(os.getenv("MAX_IMAGE_SIZE", "1600"))
    MIN_IMAGE_SIZE: int = 100
    ALLOWED_IMAGE_EXTENSIONS: set = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}

    # Gaussian Splatting training
    TRAINING_ITERATIONS: int = int(os.getenv("TRAINING_ITERATIONS", "10000"))
    SAVE_ITERATIONS: list = [10000]
    DENSIFY_UNTIL_ITER: int = 5000
    DENSIFICATION_INTERVAL: int = 200
    OPACITY_RESET_INTERVAL: int = 10000

    # COLMAP settings
    COLMAP_MAX_FEATURES: int = int(os.getenv("COLMAP_MAX_FEATURES", "16384"))
    COLMAP_NUM_THREADS: int = int(os.getenv("COLMAP_NUM_THREADS", "8"))
    COLMAP_CAMERA_MODEL: str = "OPENCV"

    # Outlier filtering
    OUTLIER_K_NEIGHBORS: int = 20
    OUTLIER_STD_THRESHOLD: float = 2.0
    OUTLIER_REMOVE_SMALL_CLUSTERS: bool = True
    OUTLIER_MIN_CLUSTER_RATIO: float = 0.01

    # Conda environment
    CONDA_ENV_NAME: str = "codyssey"
    CONDA_PYTHON: Path = Path.home() / "miniconda3" / "envs" / CONDA_ENV_NAME / "bin" / "python"

    # GPU monitoring
    GPU_CHECK_INTERVAL: int = 20  # Check every N read cycles
    MONITOR_GPU: bool = True

    # Logging
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    @classmethod
    def ensure_dirs(cls):
        """Ensure all required directories exist"""
        cls.DATA_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)
        cls.STATIC_DIR.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()

# Ensure directories exist on import
settings.ensure_dirs()
