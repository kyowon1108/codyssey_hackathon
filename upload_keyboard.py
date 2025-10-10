#!/usr/bin/env python3
import requests
from pathlib import Path
import glob

# API endpoint
API_URL = "http://localhost:8000/recon/jobs"

# Find all keyboard images
keyboard_dir = Path("/opt/ftp/files/keyboard")
image_files = sorted(glob.glob(str(keyboard_dir / "*.JPG")))

print(f"Found {len(image_files)} keyboard images")

# Prepare files for upload
files = []
for img_path in image_files:
    img_name = Path(img_path).name
    files.append(('files', (img_name, open(img_path, 'rb'), 'image/jpeg')))
    print(f"  - {img_name}")

# Upload images and create job
print("\nUploading images to API...")
response = requests.post(API_URL, files=files)

# Close file handles
for _, (_, file_obj, _) in files:
    file_obj.close()

if response.status_code == 200:
    result = response.json()
    print(f"\n✓ Job created successfully!")
    print(f"  Job ID: {result['job_id']}")
    print(f"  Public Key: {result['pub_key']}")
    print(f"\n  Viewer URL: http://localhost:8000/v/{result['pub_key']}")
    print(f"  Status URL: http://localhost:8000/recon/jobs/{result['job_id']}/status")
else:
    print(f"\n✗ Error: {response.status_code}")
    print(response.text)
