#!/usr/bin/env python3
"""
임시로 job 정보를 서버에 등록하는 스크립트
"""
import requests

# 060b81ca job 정보를 서버에 등록
url = "http://localhost:8000/admin/restore-job"
data = {
    "job_id": "060b81ca",
    "pub_key": "8296653070",
    "status": "DONE"
}

response = requests.post(url, json=data)
print(f"Status: {response.status_code}")
print(f"Response: {response.text}")
