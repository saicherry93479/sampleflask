import os
import subprocess
import numpy as np
from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
from keras import backend as K

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Download the model if not exists
if not os.path.isfile('model.h5'):
    subprocess.run(['curl', '--output', 'model.h5', 'https://media.githubusercontent.com/media/saicherry93479/sampleFlask/main/vgg.h5'], shell=True)

# Load the model
model = load_model('model.h5')

# Function to preprocess the image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Assuming input shape of your model is (224, 224)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to make predictions
def predict_image(img_path):
    img = preprocess_image(img_path)
    prediction = model.predict(img)
    return prediction

# API endpoint to accept image and return predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    # Save the uploaded file
    file_path = './uploaded_image.jpg'
    file.save(file_path)

    # Make prediction
    prediction = predict_image(file_path)

    # Return prediction
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0')
