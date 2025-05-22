from fastapi import FastAPI
import torch
import numpy as np
import sys
import os
import pickle
import logging
from schemas import InputData, OutputData
from fastapi.responses import HTMLResponse
from mangum import Mangum
from sklearn.preprocessing import MinMaxScaler
from model import create_model

# Logger per CloudWatch
logger = logging.getLogger()
logger.setLevel(logging.INFO)

app = FastAPI()

# === MODELLO CARICATO UNA SOLA VOLTA ===
MODEL_PATH = "diabetes_model.pth"
model = create_model(input_size=10)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
    model.eval()
    logger.info("✅ Modello caricato con successo.")
except Exception as e:
    logger.error(f"❌ Errore durante il caricamento del modello: {e}")
    raise e


def preprocess_input(input_data: InputData):
    smoking_map = {
        'No Info': 0, 'never': 1, 'former': 2,
        'not current': 3, 'current': 4, 'ever': 5
    }
    smoking_value = smoking_map.get(input_data.smoking_history, 0)

    gender_female = 1 if input_data.gender == "Female" else 0
    gender_male = 1 if input_data.gender == "Male" else 0
    gender_other = 1 if input_data.gender == "Other" else 0

    features = np.array([
        input_data.age,
        input_data.hypertension,
        input_data.heart_disease,
        input_data.bmi,
        input_data.hba1c_level,
        input_data.blood_glucose_level,
        smoking_value,
        gender_female,
        gender_male,
        gender_other
    ], dtype=np.float32).reshape(1, -1)

    try:
        with open('column_transformer.pkl', 'rb') as f:
            column_transformer = pickle.load(f)
        scaled_features = column_transformer.transform(features)
        return torch.FloatTensor(scaled_features)
    except FileNotFoundError:
        logger.warning("❗ Scaler non trovato. Uso scaling manuale.")
        features = features.flatten()
        continuous_indices = [0, 3, 4, 5, 6]
        continuous_vars = features[continuous_indices].reshape(1, -1)
        mm_scaler = MinMaxScaler()
        scaled_continuous = mm_scaler.fit_transform(continuous_vars).flatten()
        for i, idx in enumerate(continuous_indices):
            features[idx] = scaled_continuous[i]
        return torch.FloatTensor(features).unsqueeze(0)


@app.get("/", response_class=HTMLResponse)
async def root():
    return """
            <h1>Diabetes Prediction API</h1>
            <form id="predictionForm" onsubmit="makePrediction(event)">
                <div class="form-group">
                    <label for="age">Age:</label>
                    <input type="number" id="age" name="age" step="0.1" required>
                </div>
                
                <div class="form-group">
                    <label for="gender">Gender:</label>
                    <select id="gender" name="gender" required>
                        <option value="Female">Female</option>
                        <option value="Male">Male</option>
                        <option value="Other">Other</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="bmi">BMI:</label>
                    <input type="number" id="bmi" name="bmi" step="0.1" required>
                </div>
                
                <div class="form-group">
                    <label for="smoking_history">Smoking History:</label>
                    <select id="smoking_history" name="smoking_history" required>
                        <option value="never">Never</option>
                        <option value="former">Former</option>
                        <option value="not current">Not Current</option>
                        <option value="current">Current</option>
                        <option value="ever">Ever</option>
                        <option value="No Info">No Info</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="hypertension">Hypertension:</label>
                    <select id="hypertension" name="hypertension" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="heart_disease">Heart Disease:</label>
                    <select id="heart_disease" name="heart_disease" required>
                        <option value="0">No</option>
                        <option value="1">Yes</option>
                    </select>
                </div>
                
                <div class="form-group">
                    <label for="hba1c_level">HbA1c Level:</label>
                    <input type="number" id="hba1c_level" name="hba1c_level" step="0.1" required>
                </div>
                
                <div class="form-group">
                    <label for="blood_glucose_level">Blood Glucose Level:</label>
                    <input type="number" id="blood_glucose_level" name="blood_glucose_level" step="0.1" required>
                </div>
                
                <button type="submit">Predict</button>
            </form>
            
            <div id="result" style="margin-top: 20px; display: none;">
                <h2>Prediction Result:</h2>
                <p>Diabetes: <span id="diabetesResult"></span></p>
                <p>Probability: <span id="probabilityResult"></span></p>
            </div>

            <style>
                .form-group {
                    margin-bottom: 15px;
                }
                label {
                    display: inline-block;
                    width: 150px;
                }
                input, select {
                    padding: 5px;
                    width: 200px;
                }
                button {
                    padding: 10px 20px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                button:hover {
                    background-color: #45a049;
                }
            </style>

            <script>
                async function makePrediction(event) {
                    event.preventDefault();
                    
                    const formData = {
                        age: parseFloat(document.getElementById('age').value),
                        gender: document.getElementById('gender').value,
                        bmi: parseFloat(document.getElementById('bmi').value),
                        smoking_history: document.getElementById('smoking_history').value,
                        hypertension: parseInt(document.getElementById('hypertension').value),
                        heart_disease: parseInt(document.getElementById('heart_disease').value),
                        hba1c_level: parseFloat(document.getElementById('hba1c_level').value),
                        blood_glucose_level: parseFloat(document.getElementById('blood_glucose_level').value)
                    };

                    try {
                        const response = await fetch('/predict', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify(formData)
                        });

                        const result = await response.json();
                        
                        document.getElementById('diabetesResult').textContent = result.diabetes;
                        document.getElementById('probabilityResult').textContent = 
                            (result.probability * 100).toFixed(2) + '%';
                        document.getElementById('result').style.display = 'block';
                    } catch (error) {
                        console.error('Error:', error);
                        alert('An error occurred while making the prediction.');
                    }
                }
            </script>
        """

@app.post("/predict", response_model=OutputData)
async def predict(input_data: InputData):
    logger.info(f"📨 Ricevuto input: {input_data}")
    threshold = 0.35
    processed_input = preprocess_input(input_data)

    with torch.no_grad():
        raw_prediction = model(processed_input)
        probability = torch.sigmoid(raw_prediction)

    prob_value = float(probability[0][0])
    logger.info(f"📈 Probabilità calcolata: {prob_value:.4f}")

    return OutputData(
        diabetes="Yes" if prob_value > threshold else "No",
        probability=prob_value
    )