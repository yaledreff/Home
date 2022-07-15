
import os
import sys
import inspect

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import *
from keras import backend as K

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from model.deeplabv3p import Deeplabv3

def getCoucou():
    print('coucou')

def getDeeplabV3(n_classes):
    # Chargement du modèle de base (avec 21 classes en sorties)
    model = Deeplabv3(weights=None, input_tensor=None, infer=False,
                      input_shape=(512, 512, 3), classes=8,
                      backbone='mobilenetv2', OS=16, alpha=1)

    # On retire les dernières couches, formattées pour 21 catégories
    base_model = Model(model.input, model.layers[-4].output)

    x = Conv2D(n_classes, (1, 1), padding='same', name='conv_upsample')(base_model.layers[-2].output)
    x = Lambda(lambda x: K.tf.image.resize(x, size=(512, 512)))(x)
    x = Reshape((512 * 512, -1))(x)
    x = Activation('softmax', name='pred_mask')(x)

    return Model(base_model.input, x, name='deeplabv3p')