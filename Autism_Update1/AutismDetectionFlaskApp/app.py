from __future__ import division, print_function

from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.models import Sequential

import cv2
import os
import numpy as np
from PIL import Image as pil_image

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.layers import GlobalAveragePooling2D,Dropout,Dense

app = Flask(__name__)

def model_predict(img_path):

    model = Xception()
    new_model = Sequential()
    new_model.add(model)
    model = new_model
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.load_weights("XceptionNet.h5")

    img = cv2.resize(cv2.imread(img_path), (299, 299))
    image = np.expand_dims(img, axis=0)
    prediction_class = model.predict(image)
    prediction_class = np.argmax(prediction_class, axis=1)

    label = ""
    if prediction_class[0] == 0:
        label = 'Autistic'
    elif prediction_class[0] == 1:
        label = 'Non_Autistic'

    return label

@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':

        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        return model_predict(file_path)

    return None

if __name__ == '__main__':
    app.run(debug=True)

