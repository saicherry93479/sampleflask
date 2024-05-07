import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import keras
from keras.models import load_model
# from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K

from flask import Flask

app = Flask(__name__)

import os
import subprocess

if not os.path.isfile('model.h5'):
    subprocess.run(['curl', '--output', 'model.h5', 'https://media.githubusercontent.com/media/saicherry93479/sampleFlask/main/vgg.h5'], shell=True)


print("came one ")

# K.clear_session()
model = load_model('model.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming input shape of your model is (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img_path):
    print("hii in ")
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    return prediction

prediction=predict_image('./sample.jpeg')
# print(prediction)
@app.route('/')
def home():
    prediction=predict_image('./sample.jpeg')
    return 'Hello, this is your Flask app running on Google Colab!'


if __name__ == '__main__':
    app.run(host='0.0.0.0')