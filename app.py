from flask import Flask, render_template, request
import pickle
import numpy as np
from utils import preprocess_data

app = Flask(__name__)

# Load the trained model
with open('heart_model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form[feature]) for feature in ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']]
    input_data = np.array([features])
    input_data = preprocess_data(input_data)  # You need to define this function in utils.py
    prediction = model.predict(input_data)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8989, debug=False)

