from fastapi import FastAPI
from pydantic import BaseModel
import mlflow
import numpy as np

app = FastAPI(title="Fraud Detection API")

# Load model from MLflow Registry
MODEL_NAME = "fraud-model"
MODEL_STAGE = "Production"   # you set your baseline model to Production

model = mlflow.pyfunc.load_model(
    model_uri=f"models:/{MODEL_NAME}/{MODEL_STAGE}"
)

class PredictionInput(BaseModel):
    features: list[float]

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(input_data: PredictionInput):
    data = np.array(input_data.features).reshape(1, -1)

    preds = model.predict(data)

    # assuming your model returns 0/1 only
    fraud_label = int(preds[0])

    # probability: if model does not return probas, fallback
    try:
        probas = model.predict_proba(data)
        fraud_prob = float(probas[0][1])
    except:
        fraud_prob = fraud_label * 1.0

    return {
        "fraud_probability": fraud_prob,
        "fraud_label": fraud_label
    }
