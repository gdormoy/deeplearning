import os
import cv2
import numpy as np
from tensorflow.python import keras
from tensorflow.python.keras.models import *

def load_existing_model(model_filename):
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    return model

if __name__ == "__main__":
    model = load_existing_model('./models/project.h5')
    model.summary()
    file = input('path to the image!\n')
    img = cv2.imread(file)
    img = cv2.resize(img, (128,128))
    image = np.array(img).reshape(-1,128,128,3)
    result = model.predict(image)
    print("{}: {}".format(file, result))
