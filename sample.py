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


print("came one ")

# K.clear_session()
model = load_model('vgg.h5')

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
    prediction=predict_image('/content/drive/My Drive/samplemodel.jpeg')
    return 'Hello, this is your Flask app running on Google Colab!'


if __name__ == '__main__':
    app.run()