from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
from PIL import Image
import os

app = Flask(__name__)

# Load model once
model = tf.keras.models.load_model('CNN_Fruits_Classification.h5')

Classes = ['apple', 'banana', 'orange']


def preprocess_image(img):
    img = img.convert("RGB")
    img = img.resize((32, 32))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    image_path = None
    probabilities = None

    if request.method == "POST":
        if "image" not in request.files:
            return render_template("index.html", error="No file uploaded")

        file = request.files["image"]

        if file.filename == "":
            return render_template("index.html", error="No selected file")

        image_path = os.path.join("static", file.filename)
        file.save(image_path)

        # Process + Predict
        img = Image.open(image_path)
        img_array = preprocess_image(img)
        preds = model.predict(img_array)

        predicted_class = Classes[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100
        probabilities = {Classes[i]: float(preds[0][i]) * 100 for i in range(3)}

        prediction_result = {
            "class": predicted_class,
            "confidence": confidence,
        }

    return render_template(
        "index.html",
        prediction=prediction_result,
        image_path=image_path,
        probabilities=probabilities
    )


if __name__ == "__main__":
    app.run(debug=True)
