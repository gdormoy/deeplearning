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
from tensorflow.python.keras.callbacks import *
from tensorflow.python.keras.utils import to_categorical

train_data = "./dataset/training"
test_data = "./dataset/testing"

def one_hot_label(img):
    label = img.split(' ')[0]
    if 'pepe' in img:
        ohl = np.array([1, 0])
    else:
        ohl = np.array([0, 1])
    return ohl

def train_data_with_label():
    train_images = []
    for i in tqdm(os.listdir(train_data)):
        path = os.path.join(train_data, i)
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128))
        train_images.append([np.array(img), one_hot_label(i)])
    shuffle(train_images)
    return train_images

def test_data_with_label():
    test_images = []
    for i in tqdm(os.listdir(test_data)):
        path = os.path.join(test_data, i)
        img = cv2.imread(path)
        img = cv2.resize(img, (128,128))
        test_images.append([np.array(img), one_hot_label(i)])
    shuffle(test_images)
    return test_images


if __name__ == "__main__":
    training_images = train_data_with_label()
    testing_images = test_data_with_label()
    tr_img_data = np.array([i[0] for i in training_images]).reshape(-1,128,128,3)
    tr_lbl_data = np.array([i[1] for i in training_images])
    tst_img_data = np.array([i[0] for i in testing_images]).reshape(-1,128,128,3)
    tst_lbl_data = np.array([i[1] for i in testing_images])

    model = Sequential()
    tb_callback = TensorBoard('./logs/project')
    # model.add(InputLayer(input_shape=[128,128,3]))
    model.add(Conv2D(filters=32,kernel_size=5,strides=1,padding='same', activation='relu', input_shape=[128,128,3]))
    model.add(MaxPool2D(pool_size=5,padding='same'))

    model.add(Conv2D(filters=64, kernel_size=5,strides=1, padding='same', activation='relu', input_shape=[128,128,3]))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Conv2D(filters=128, kernel_size=5,strides=1, padding='same', activation='relu', input_shape=[128,128,3]))
    model.add(MaxPool2D(pool_size=5, padding='same'))

    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(rate=0.5))
    model.add(Dense(2,activation='softmax'))
    optimizer = Adam(lr=1e-3)

    model.compile(optimizer=optimizer,loss='categorical_crossentropy',metrics=['accuracy'])
    model.fit(x=tr_img_data,y=tr_lbl_data,epochs=50,batch_size=100, callbacks=[tb_callback])
    score = model.evaluate(x=tst_img_data,y=tst_lbl_data,batch_size=100, callbacks=[tb_callback])

    print("%s: %.2f%%" % (model.metrics_names[1], score[1]*100))

    model.save('./models/project.h5')
    json_string = model.to_json()
    with open("./models/project.json", "w") as json_file:
        json_file.write(json_string)
