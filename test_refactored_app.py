"""
Test script for refactored application
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test all module imports"""
    print("Testing module imports...")

    try:
        from app.config import settings
        print(" Config imported")

        from app.db.models import Job, ErrorLog
        print(" Database models imported")

        from app.db.database import init_db, get_db
        print(" Database utilities imported")

        from app.db import crud
        print(" CRUD operations imported")

        from app.schemas.job import JobCreateResponse, JobStatusResponse
        print(" Schemas imported")

        from app.utils.logger import setup_logger
        print(" Logger imported")

        from app.utils.system import get_gpu_memory_usage
        print(" System utilities imported")

        from app.utils.image import validate_image_file, save_image
        print(" Image utilities imported")

        from app.core.pipeline import run_command
        print(" Pipeline imported")

        from app.core.colmap import COLMAPPipeline
        print(" COLMAP pipeline imported")

        from app.core.gaussian_splatting import GaussianSplattingTrainer
        print(" Gaussian Splatting trainer imported")

        from app.api import jobs, viewer
        print(" API routers imported")

        from app.main import app
        print(" Main app imported")

        return True
    except Exception as e:
        print(f" Import failed: {e}")
        return False


def test_database():
    """Test database initialization"""
    print("\nTesting database...")

    try:
        from app.db.database import init_db
        from app.config import settings

        # Initialize database
        init_db()
        print(f" Database initialized at {settings.DATABASE_URL}")

        return True
    except Exception as e:
        print(f" Database test failed: {e}")
        return False


def test_config():
    """Test configuration"""
    print("\nTesting configuration...")

    try:
        from app.config import settings

        print(f"  BASE_DIR: {settings.BASE_DIR}")
        print(f"  DATA_DIR: {settings.DATA_DIR}")
        print(f"  HOST: {settings.HOST}")
        print(f"  PORT: {settings.PORT}")
        print(f"  MAX_CONCURRENT_JOBS: {settings.MAX_CONCURRENT_JOBS}")
        print(f"  TRAINING_ITERATIONS: {settings.TRAINING_ITERATIONS}")
        print(" Configuration loaded")

        return True
    except Exception as e:
        print(f" Config test failed: {e}")
        return False


def test_app_routes():
    """Test FastAPI app routes"""
    print("\nTesting app routes...")

    try:
        from app.main import app

        routes = [route.path for route in app.routes]
        print(f"  Found {len(routes)} routes:")
        for route in routes:
            print(f"    - {route}")

        print(" Routes registered")
        return True
    except Exception as e:
        print(f" Routes test failed: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Refactored Application Test Suite")
    print("=" * 60)

    results = []

    results.append(("Imports", test_imports()))
    results.append(("Config", test_config()))
    results.append(("Database", test_database()))
    results.append(("Routes", test_app_routes()))

    print("\n" + "=" * 60)
    print("Test Results:")
    print("=" * 60)

    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"{name}: {status}")

    all_passed = all(result for _, result in results)
    print("=" * 60)

    if all_passed:
        print(" All tests passed!")
        sys.exit(0)
    else:
        print(" Some tests failed")
        sys.exit(1)
