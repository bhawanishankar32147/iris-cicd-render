# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return "Iris Classifier API is Running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from the request
        data = request.get_json(force=True)
        
        # Validate input: ensure 'features' key exists and contains 4 numerical values
        if "features" not in data or len(data["features"]) != 4:
            return jsonify({"error": "Exactly 4 numerical features are required"}), 400
        
        # Convert features to NumPy array and reshape for prediction
        features = np.array(data["features"], dtype=float).reshape(1, -1)
        
        # Make prediction
        prediction = model.predict(features)[0]
        
        # Map numerical prediction to class name
        classes = ["setosa", "versicolor", "virginica"]
        result = {"prediction": classes[prediction]}
        
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)