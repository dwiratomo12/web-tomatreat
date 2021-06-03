import base64
import numpy as np
import io
from PIL import Image
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, load_model
from keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from flask import request
from flask import jsonify
from flask import Flask
import json as json
from flask_cors import CORS, cross_origin

app = Flask(__name__)
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'

def get_model():
    global model
    model = load_model('model70epoch.hdf5')
    print(" * Model loaded!")

def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image/255
    image = np.expand_dims(image, axis=0)
    
    return image
    
print("* Loading Keras model....")
get_model()

@app.route("/leafpredict", methods=["POST"])
@cross_origin()
def predict():
    message = request.get_json(force=True)
    encoded = message['image']
    decoded = base64.b64decode(encoded)
    image = Image.open(io.BytesIO(decoded))
    processed_image = preprocess_image(image, target_size=(256, 256))

    prediction = model.predict(processed_image).tolist()

    response = {
        'prediction': {
            'satu': prediction[0][0],
            'dua': prediction[0][1],
            'tiga': prediction[0][2],
            'empat': prediction[0][3],
            'lima': prediction[0][4],
            'enam': prediction[0][5],
            'tujuh': prediction[0][6],
            'delapan': prediction[0][7],
            'sembilan': prediction[0][8],
            'sepuluh': prediction[0][9]
        }
    }
    return jsonify(response)