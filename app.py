import os
import requests
from flask import Flask, request, render_template, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

# Suppress TensorFlow oneDNN warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)  # Generate a random secret key
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Loading the pre-trained model
model_path = 'saved_model_iteration_final_1/animal_identifier_trained_model_final_1.h5'
cnn_model = load_model(model_path)

# Defining class labels
class_labels = ['cat', 'dog', 'elephant', 'horse', 'lion'] 

# Placeholder accuracy score for the model
accuracy_score = 0.8663

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(img, target_size=(50, 50)):
    # Convert to RGB if the image has an alpha channel
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize the image
    return img_array

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files and 'url' not in request.form:
            flash('No file part or URL provided')
            return redirect(request.url)

        # Handle file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                img = Image.open(file_path)
                img_array = preprocess_image(img)
                predictions = cnn_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                prediction_label = class_labels[predicted_class[0]]
                # Pass the URL for the uploaded file
                img_url = url_for('uploaded_file', filename=filename)
                return render_template('result.html', label=prediction_label, img_path=img_url, accuracy=accuracy_score)

        # Handle URL submission
        if 'url' in request.form:
            url = request.form['url']
            try:
                headers = {'User-Agent': 'Mozilla/5.0'}
                response = requests.get(url, headers=headers)
                response.raise_for_status()
                img = Image.open(BytesIO(response.content))
                img_array = preprocess_image(img)
                predictions = cnn_model.predict(img_array)
                predicted_class = np.argmax(predictions, axis=1)
                prediction_label = class_labels[predicted_class[0]]
                img_path = url  # Use the URL directly for display
                return render_template('result.html', label=prediction_label, img_path=img_path, accuracy=accuracy_score)
            except requests.exceptions.RequestException as e:
                flash(f"Error fetching the image from URL: {e}")
                return redirect(request.url)

    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)