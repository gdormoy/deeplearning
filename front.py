from flask import Flask
from flask import render_template
import cv2
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from flask import request

app = Flask(__name__)


def load_image(img_path, show=False):
    img = image.load_img(img_path, target_size=(128, 128))
    img_tensor = image.img_to_array(img)          # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    img_tensor /= 255.                                      # imshow expects values in the range [0, 1]
    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/evaluate", methods = ['POST'])
def evaluate():
     request_file = request.files['file']
     model = load_model("./models/project.h5")
     img = load_image(request_file)
     pred = model.predict(img)
     return "This is Pepe the Frog at {:f}%".format(pred[0][1] * 100)


if __name__ == "__main__":
    app.run(debug=True)
