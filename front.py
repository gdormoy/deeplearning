from flask import Flask
from flask import render_template
from tensorflow.python.keras.models import load_model
import numpy as np
from flask import request
from skimage import transform
from PIL import Image

app = Flask(__name__)


def load_image(img_path):
    np_image = Image.open(img_path)
    np_image = np.array(np_image).astype('float32')
    np_image = transform.resize(np_image, (128, 128, 3))
    return np.expand_dims(np_image, axis=0)


@app.route("/")
def home():
    return render_template('index.html')


@app.route("/evaluate", methods=['POST'])
def evaluate():
    request_file = request.files['file']
    img = load_image(request_file)
    model = load_model("./models/project.h5")
    pred = model.predict(img)
    return "This is Pepe the Frog at {:f}%".format(pred[0][0] * 100)


if __name__ == "__main__":
    app.run(debug=True)
