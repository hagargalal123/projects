from flask import Flask, render_template, request
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread 
from skimage.transform import resize
from sklearn import svm
import pickle

app = Flask(__name__)

# Load the trained model
model = pickle.load(open('img_svc_model1.p', 'rb'))

# Define image categories
CATEGORIES = ['pretty sunflower', 'rugby ball leather', 'ice cream cone']

@app.route('/')
def home():
    return render_template('index_image.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        url = request.form['url']
        img = imread(url)
        img_resized = resize(img, (150, 150, 3))
        flat_data_test = np.array([img_resized.flatten()])
        y_out = model.predict(flat_data_test)
        prediction = CATEGORIES[y_out[0]]
        return render_template('index_image.html', prediction=prediction, url=url)
    except Exception as e:
        return render_template('index_image.html', error="Error processing the image. Please try again.", url=url)

if __name__ == '__main__':
    app.run(debug=True)
