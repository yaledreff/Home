# This is a sample Python script.

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import uvicorn
from PIL import Image
from fastapi import FastAPI
from fastapi import File
from fastapi import UploadFile


from starlette.responses import StreamingResponse
from utils import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

@app.post("/mask")
def get_mask(file: UploadFile = File(...)):
    # chargement de l'image
    imageSrc = Image.open(file.file)
    imageRes = processImage(imageSrc)
    # Reconstitution d'une image en RGB pour affectation de couleurs facilement visibles à l'écran
    imagePred3D = np.stack((imageRes, imageRes, imageRes), axis=2)
    imagePredColored = getColoredMask(imagePred3D)
    imagePredColored = np.array(imagePredColored, dtype='uint8')
    imagePredColored = cv2.resize(imagePredColored, (512, 256), interpolation=cv2.INTER_AREA)
    return StreamingResponse(getBufferImage(Image.fromarray(imagePredColored, 'RGB')), media_type="image/jpeg")

@app.post("/boxes")
def get_boxes(file: UploadFile = File(...)):
    # chargement de l'image
    imageSrc = Image.open(file.file)
    imageMaskRes = processImage(imageSrc)
    imageSrc = cv2.resize(np.asarray(imageSrc), (512, 512))
    imageMaskRes = cv2.resize(imageMaskRes, (512, 512))
    imageWithBoxes = imageBoxesDbg(imageSrc, imageMaskRes, alpha=0.4)
    return StreamingResponse(getBufferImage(Image.fromarray(imageWithBoxes, 'RGB')), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run("main:app")
