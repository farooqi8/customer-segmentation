from fastapi import APIRouter
import joblib
import pandas as pd
from api.schemas import CustomerInput, PredictionOutput

router = APIRouter()

model = joblib.load('models/kmeans_model.joblib')
scaler = joblib.load('models/scaler.joblib')

labels = {
    0: 'High Value Active Customers',
    1: 'Window Shoppers',
    2: 'Premium Rare Buyers',
    3: 'Casual Low Spenders'
}

@router.post('/predict', response_model=PredictionOutput)
def predict_segment(customer: CustomerInput):
    data = customer.model_dump()
    df = pd.DataFrame([data])
    scaled = scaler.transform(df)
    cluster = model.predict(scaled)[0]
    segment = labels[cluster]
    return PredictionOutput(cluster=int(cluster), segment=segment)