from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import h5py

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model
model = tf.keras.models.load_model('model.h5')

# Dog breed labels
dog_classes = [
    'affenpinscher', 'beagle', 'appenzeller', 'basset', 'bluetick', 'boxer', 'cairn',
    'doberman', 'german_shepherd', 'golden_retriever', 'malamute', 'pug', 
    'saint_bernard', 'scottish_deerhound', 'shih_tzu', 'staffordshire_bullterrier', 
    'wheaten_terrier', 'yorkshire_terrier'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

@app.route('/output', methods=['POST'])
def output():
    if request.method == 'POST':
        # Save the uploaded file
        f = request.files['file']
        filename = secure_filename(f.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        f.save(filepath)

        # Load and preprocess the image
        img = load_img(filepath, target_size=(220, 220))  # Adjust target size as per your model
        image_array = img_to_array(img) / 255.0  # Normalize the image
        image_array = np.expand_dims(image_array, axis=0)

        # Make prediction
        preds = np.argmax(model.predict(image_array), axis=1)
        prediction = dog_classes[int(preds)]
        
        return render_template('output.html', prediction=prediction)
    return redirect(url_for('predict'))

if __name__ == '__main__':
    app.run(debug=True)
