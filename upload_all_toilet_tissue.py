#!/usr/bin/env python3
import requests
from pathlib import Path

# 모든 toilet tissue 이미지 찾기
image_dir = Path("/home/kapr/Desktop/toilet_tissue/images")
images = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.png")) + list(image_dir.glob("*.PNG"))

print(f"Found {len(images)} images in {image_dir}")

# 모든 이미지를 업로드
files = []
for img_path in sorted(images):
    files.append(('files', (img_path.name, open(img_path, 'rb'), 'image/jpeg')))

print(f"Uploading {len(files)} images...")
response = requests.post('http://localhost:8000/recon/jobs', files=files)

# 파일 핸들 닫기
for _, (_, f, _) in files:
    f.close()

if response.status_code == 200:
    data = response.json()
    print(f"Job created successfully!")
    print(f"Job ID: {data['job_id']}")
    print(f"Public Key: {data['pub_key']}")
    print(f"Viewer URL: http://localhost:8000/v/{data['pub_key']}")
    print(f"Status URL: http://localhost:8000/recon/jobs/{data['job_id']}/status")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
