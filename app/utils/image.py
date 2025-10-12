"""
Image processing and validation utilities
"""
from pathlib import Path
from PIL import Image
from fastapi import UploadFile, HTTPException
import io
from app.config import settings
from app.utils.logger import setup_logger

logger = setup_logger(__name__)


def validate_image_file(file: UploadFile) -> None:
    """
    Validate uploaded image file

    Args:
        file: Uploaded file from FastAPI

    Raises:
        HTTPException: If file is invalid
    """
    # Check file extension
    file_ext = Path(file.filename).suffix
    if file_ext not in settings.ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(
            400,
            f"Unsupported file type: {file_ext}. Allowed: {settings.ALLOWED_IMAGE_EXTENSIONS}"
        )

    # Validate file content
    try:
        content = file.file.read()
        img = Image.open(io.BytesIO(content))
        img.verify()

        # Reset file pointer
        file.file.seek(0)

        # Check minimum size
        if img.width < settings.MIN_IMAGE_SIZE or img.height < settings.MIN_IMAGE_SIZE:
            raise HTTPException(
                400,
                f"Image too small: {img.width}x{img.height}. Minimum: {settings.MIN_IMAGE_SIZE}x{settings.MIN_IMAGE_SIZE}"
            )

        logger.info(f"Validated image: {file.filename} ({img.width}x{img.height})")

    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(400, f"Invalid image file: {file.filename}. Error: {str(e)}")


async def save_image(file: UploadFile, output_path: Path, resize: bool = True) -> None:
    """
    Save uploaded image with optional resizing

    Args:
        file: Uploaded file from FastAPI
        output_path: Output file path
        resize: Whether to resize image
    """
    content = await file.read()

    if not resize:
        # Save original
        with open(output_path, 'wb') as f:
            f.write(content)
        return

    # Resize and save
    try:
        img = Image.open(io.BytesIO(content))
        width, height = img.size

        if width > settings.MAX_IMAGE_SIZE or height > settings.MAX_IMAGE_SIZE:
            ratio = min(settings.MAX_IMAGE_SIZE / width, settings.MAX_IMAGE_SIZE / height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)
            img = img.resize((new_width, new_height), Image.LANCZOS)

        img.save(output_path, quality=95)
        logger.info(f"Saved image: {output_path.name}")

    except Exception as e:
        logger.warning(f"Failed to resize image {file.filename}: {e}. Saving original.")
        with open(output_path, 'wb') as f:
            f.write(content)
