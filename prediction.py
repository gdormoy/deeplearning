import os
import cv2
import numpy as np
import tensorflow as tf
from random import shuffle
from tqdm import tqdm
from tensorflow.python import keras
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.datasets import *
from tensorflow.python.keras.activations import *
from tensorflow.python.keras.optimizers import *
from tensorflow.python.keras.losses import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.utils import to_categorical

def load_existing_model(model_filename):
    if os.path.exists(model_filename):
        model = load_model(model_filename)
    return model

if __name__ == "__main__":
    model_filename = input('path to the saved model!\n')
    model = load_existing_model(model_filename)
    model.summary()
    file = input('path to the image!\n')
    img = cv2.imread(file)
    img = cv2.resize(img, (128,128))
    image = np.array(img).reshape(-1,128,128,3)
    result = model.predict(image)
    print("{}   {}".format(file, result))

    # predicting_images = prediction(dataset,model)
    # pr_img_data = np.array([i[0] for i in predict_images]).reshape(-1,128,128,3)
    # pr_lbl_data = np.array([i[1] for i in predict_images])
    #
    # score = model.evaluate(X=pr_img_data, Y=pr_lbl_data, verbose=0)
    # print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))
