import pytest
import requests

BASE_URL = "http://127.0.0.1:8000"


def test_upload_file_success():
    file_content = b"Artificial Intelligence is transformative."
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": ("sample.txt", file_content := file_content, "text/plain")},
    )
    assert response.status_code == 201
    json_resp = response.json()
    assert json_resp["status"] == "success"
    assert "filename" in json_resp


def test_upload_invalid_file_type():
    response = requests.post(
        f"{BASE_URL}/upload",
        files={"file": ("fake.jpg", b"fakecontent", "image/jpeg")},
    )
    assert response.status_code == 400
    json_resp = response.json()
    assert "Invalid file type" in json_resp["detail"]


def test_search_query_success():
    response = requests.get(f"{BASE_URL}/search?query=What is AI?")
    assert response.status_code == 200
    json_resp = response.json()
    assert "answer" in json_resp
    assert isinstance(json_resp["answer"], str)


def test_search_query_no_query():
    response = requests.get(f"{BASE_URL}/search")
    assert response.status_code == 422


def test_static_file_serving():
    response = requests.get(f"{BASE_URL}/static/images/doctalk.jpg")
    assert response.status_code == 200
    assert response.headers["content-type"] in ["image/jpeg", "image/jpg"]
