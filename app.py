# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("models/final_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get inputs from form
    features = [float(x) for x in request.form.values()]
    final_features = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(final_features)[0]

    return render_template("index.html", prediction_text=f"Predicted Class: {prediction}")

if __name__ == "__main__":
    app.run(debug=True)
