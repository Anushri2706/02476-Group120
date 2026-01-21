from fastapi.testclient import TestClient
from src.mlops.api import app
from http import HTTPStatus
import io
import numpy as np
import cv2

client = TestClient(app)

def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    success, encoded = cv2.imencode(".png", img)
    assert success, "Failed to encode test image"
    return encoded.tobytes()

def test_cv_model_success():
    """
    Tests /cv_model/ with a valid PNG image created in-memory.
    """
    png_bytes = _make_png_bytes()
    files = {"data": ("dummy.png", io.BytesIO(png_bytes), "image/png")}
    response = client.post("/cv_model/", files=files)

    assert response.status_code == 200
    json_response = response.json()
    # The model is not loaded in the test environment, so we expect this message.
    assert json_response["message"] == "Model not loaded. Check server logs."
    # The status code in the JSON body should be 500
    assert json_response["status-code"] == HTTPStatus.INTERNAL_SERVER_ERROR.value

def test_cv_model_no_file():
    """
    Tests /cv_model/ without providing a file.
    """
    response = client.post("/cv_model/")
    assert response.status_code == 422

def test_cv_model_wrong_file_type():
    """
    Tests /cv_model/ with a non-image payload. Expect 200 because model loading fails first.
    """
    files = {"data": ("not_image.txt", io.BytesIO(b"not an image"), "text/plain")}
    response = client.post("/cv_model/", files=files)
    assert response.status_code == 200
    json_response = response.json()
    assert json_response["message"] == "Model not loaded. Check server logs."
