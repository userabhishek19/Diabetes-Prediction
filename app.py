import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

# Load the pre-trained model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input values from the form and convert them to float
        float_features = [float(x) for x in request.form.values()]
        final_features = np.array(float_features).reshape(1, -1)

        # Scale the input features
        scaled_features = scaler.transform(final_features)

        # Make prediction
        prediction = model.predict(scaled_features)

        # Interpret the prediction
        if prediction == 1:
            pred = "You have Diabetes, please consult a Doctor."
        else:
            pred = "You don't have Diabetes."
        
        return render_template('index.html', prediction_text=pred)

    except ValueError:
        return render_template('index.html', prediction_text="Invalid input. Please enter valid numeric values.")

if __name__ == "__main__":
    app.run(debug=True)
