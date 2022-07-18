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
# import cv2

from starlette.responses import StreamingResponse
#
from MdeeplabV3 import *
from utils import *

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome from the API"}

    # return StreamingResponse(content=imgio, media_type="image/jpeg")

# if __name__ == "__main__":
#     uvicorn.run("main:app")
