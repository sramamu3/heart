# app.py
from flask import Flask, render_template, request
import pickle
import numpy as np
import os
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
@app.route('/')
def home():
    return render_template("index.html")
@app.route('/predict', methods=['POST'])
def predict():
    # Convert all form values to float
    input_data = [float(x) for x in request.form.values()]
    features = np.array([input_data])
    prediction = model.predict(features)[0]
    result = "Heart Disease Detected " if prediction == 1 else "No Heart Disease "
    return render_template("index.html", prediction_text=result)
if __name__ == "__main__":
    app.run(debug=True)
    port=int(os.environ.get("PORT",5000)) 
    app.run(host="0.0.0.0",port=port)
