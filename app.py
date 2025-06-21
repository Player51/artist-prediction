from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import preprocess_input # type:ignore

def build_exact_notebook_model():
    IMG_SIZE = 299
    NUM_CLASSES = 10 
    
    original_model = tf.keras.applications.ResNet50V2(
        include_top=False,
        weights='imagenet', 
    )
    original_model.trainable = False # As per the notebook

    # 2. Input Layer
    model_inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    # 3. Preprocessing Layer
    x = tf.keras.applications.resnet_v2.preprocess_input(model_inputs)
    x = original_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.4)(x) # Dropout rate from the notebook
    x = tf.keras.layers.Dense(128, activation='relu')(x) # Dense layer from the notebook
    model_outputs = tf.keras.layers.Dense(NUM_CLASSES, activation='softmax')(x) # Output layer

    model = tf.keras.Model(inputs=model_inputs, outputs=model_outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # Optimizer and LR from notebook
        loss='sparse_categorical_crossentropy', # Loss from notebook
        metrics=['sparse_categorical_accuracy'] # Metrics from notebook
    )

    return model

# Load model and weights
model = build_exact_notebook_model()
model.load_weights("model/model.weights.h5")  

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Replace these with actual class labels if available
CLASS_NAMES = ['Cezanne', 'Degas', 'Gauguin', 'Hassam', 'Matisse', 'Monet', 'Pissarro', 'Renoir', 'VanGogh', 'Sargent']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part", 400
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)
            predicted_artist = predict_artist(file_path)

            return render_template('result.html', artist=predicted_artist)
    return render_template("index.html")

def predict_artist(image_path):
    # Load and preprocess image
    im = Image.open(image_path).convert('RGB')
    im = im.resize((299, 299))
    im_array = np.asarray(im)
    im_array = np.expand_dims(im_array, axis=0)
    im_array = preprocess_input(im_array)

    # Predict
    prediction = model.predict(im_array)[0]
    predicted_index = np.argmax(prediction)
    predicted_label = CLASS_NAMES[predicted_index]
    confidence = prediction[predicted_index]

    return f"{predicted_label} ({confidence:.2%} confidence)"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
