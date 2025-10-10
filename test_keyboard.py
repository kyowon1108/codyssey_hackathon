#!/usr/bin/env python3
import requests
from pathlib import Path

# 키보드 이미지 디렉토리
keyboard_dir = Path("/opt/ftp/files/keyboard")

# 이미지 파일들 수집
image_files = []
for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
    image_files.extend(list(keyboard_dir.glob(ext)))

image_files = sorted(image_files)
print(f"Found {len(image_files)} images")

if not image_files:
    print("No images found!")
    exit(1)

# 서버에 업로드
files = []
for img_path in image_files:
    files.append(('files', (img_path.name, open(img_path, 'rb'), 'image/jpeg')))

print(f"Uploading {len(files)} images to server...")
response = requests.post('http://localhost:8000/recon/jobs', files=files)

# 파일 핸들 닫기
for _, (_, f, _) in files:
    f.close()

if response.status_code == 200:
    result = response.json()
    print(f"✓ Job created successfully!")
    print(f"  Job ID: {result['job_id']}")
    print(f"  Public Key: {result['pub_key']}")
    print(f"  Viewer URL: http://localhost:8000/v/{result['pub_key']}")
else:
    print(f"✗ Error: {response.status_code}")
    print(response.text)
