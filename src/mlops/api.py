from fastapi import UploadFile, File, FastAPI
from http import HTTPStatus
from typing import Optional
import cv2
import numpy as np

app = FastAPI()


@app.post("/cv_model/")
async def cv_model(data: UploadFile = File(...)):
    with open('image.jpg', 'wb') as image:
        content = await data.read()

        nparr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)


        h, w = 64, 64
        res = cv2.resize(img, (h, w))


    response = {
        "filename": data.filename,
        "original_shape": img.shape,
        "resized_shape": res.shape,
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK.value,
    }
    return response