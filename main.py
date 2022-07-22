# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import joblib
import numpy as np
import uvicorn
import os
import io
import sys
import inspect
from PIL import Image
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile
import cv2

from starlette.responses import StreamingResponse
from MdeeplabV3 import *
from utils import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/score")
def get_image(file: UploadFile = File(...)):
    # chargement de l'image
    imageSrc = Image.open(file.file)
    imageArray = np.asarray(imageSrc)
    image = cv2.resize(imageArray, (512, 512))
    image = image.reshape((1, 512, 512, 3))
    print(str(image.shape))

    # chargement du modèle de traitement (Deeplab V3) avec tous ses layers
    model = getDeeplabV3(n_classes=8)

    # chargement des poids calculés lors de l'entrainement
    model.load_weights('model/DeeplabV3WeightsRUN.h5')

    # On utilise le modèle pour prédire le mask de l'image source
    imagePred = np.argmax(model.predict(image), -1)[0].reshape((512, 512))
    # Reconstitution d'une image en RGB pour affectation de couleurs facilement visibles à l'écran
    imagePred3D = np.stack((imagePred, imagePred, imagePred), axis=2)
    imagePredColored = getColoredMask(imagePred3D)

    # imgio = io.BytesIO()
    # image = Image.fromarray(imagePredColored, 'RGB')
    # image.save(imgio, 'JPEG')
    # imgio.seek(0)
    res, im_png = cv2.imencode(".png", imagePredColored)

    return StreamingResponse(io.BytesIO(im_png.tobytes()), media_type="image/png")

# if __name__ == "__main__":
#     uvicorn.run("main:app")
