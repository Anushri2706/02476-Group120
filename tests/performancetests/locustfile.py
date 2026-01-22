import os
from io import BytesIO

import cv2
import numpy as np
from locust import HttpUser, between, task


def _make_png_bytes(width: int = 64, height: int = 64) -> bytes:
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    ok, enc = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode dummy image")
    return enc.tobytes()


class MyUser(HttpUser):
    wait_time = between(1, 3)
    host = os.getenv("TARGET_HOST", None)  # Allow CLI/env override

    @task
    def test_cv_model(self):
        png_bytes = _make_png_bytes()
        buf = BytesIO(png_bytes)
        self.client.post(
            "/cv_model/",
            files={"data": ("dummy.png", buf, "image/png")},
        )
