import cv2
import numpy as np
import skimage.morphology as mph
from skimage.measure import label, regionprops
import io
from MdeeplabV3 import *

color_map = {
 '0': [0, 0, 0],
 '1': [153, 153, 0],
 '2': [255, 204, 204],
 '3': [255, 0, 127],
 '4': [0, 255, 0],
 '5': [0, 204, 204],
 '6': [255, 0, 0],
 '7': [0, 0, 255]
}

def getColoredMask(image):
 sourcesColors = [(0, 0, 0), (1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6), (7, 7, 7)]
 targetColors = [(0, 0, 0), (153, 153, 0), (255, 204, 204), (255, 0, 127), (0, 255, 0), (0, 204, 204), (255, 0, 0), (0, 0, 255)]
 imageSAV = image

 for i in range(0, 8):
  sr, sg, sb = sourcesColors[i]
  tr, tg, tb = targetColors[i]
  red, green, blue = image[:, :, 0], image[:, :, 1], image[:, :, 2]
  mask = (red == sr) & (green == sg) & (blue == sb)
  image[:, :, :3][mask] = [tr, tg, tb]
 return image

def imageBoxes(image, maskPredict, alpha=0.5):
    def isIn(val, target):
        return target == val

    vFunc = np.vectorize(isIn)
    imageSrc = image
    imageColored = imageSrc
    overlay = imageSrc.copy()
    for i in [0, 5, 2, 1, 3, 4, 7, 6]:
        mask = vFunc(maskPredict, i)
        lbl = mph.label(mask)
        props = regionprops(lbl)
        for prop in props:
            cv2.rectangle(overlay, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), color_map[str(i)],
                      cv2.FILLED)

    return cv2.addWeighted(overlay, alpha, imageColored, 1 - alpha, 0)

def imageBoxesDbg(image, maskPredict, alpha=0.4):
    def isIn(val, target):
        return target == val

    vFunc = np.vectorize(isIn)
    imageSrc = image
    imageColored = imageSrc
    overlay = imageSrc.copy()
    for i in [0, 5, 2, 1, 3, 4, 7, 6]:
        mask = vFunc(maskPredict, i)
        lbl = mph.label(mask)
        props = regionprops(lbl)
        for prop in props:
            print(i)
            cv2.rectangle(overlay, (prop.bbox[1], prop.bbox[0]), (prop.bbox[3], prop.bbox[2]), color_map[str(i)],
                      cv2.FILLED)

    return cv2.addWeighted(overlay, alpha, imageColored, 1 - alpha, 0)

def get_bytes_value(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='JPEG')
    return img_byte_arr.getvalue()

def getBufferImage(image):
    filtered_image = io.BytesIO()
    image.save(filtered_image, "JPEG")
    filtered_image.seek(0)
    return filtered_image

def processImage(imageSrc):
    imageArray = np.asarray(imageSrc)
    image = cv2.resize(imageArray, (512, 512))
    image = image.reshape((1, 512, 512, 3))
    # chargement du modèle de traitement (Deeplab V3) avec tous ses layers
    model = getDeeplabV3(n_classes=8)
    # chargement des poids calculés lors de l'entrainement
    model.load_weights('model/DeeplabV3WeightsRUN.h5')
    # On utilise le modèle pour prédire le mask de l'image source
    imagePred = np.argmax(model.predict(image), -1)[0].reshape((512, 512))
    return imagePred