from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np


# Create FastAPI app
app = FastAPI(title="Car Price Predictor")

# Load trained model
model = joblib.load("model.pkl")

# Input schema using Pydantic
from pydantic import BaseModel

from pydantic import BaseModel

class CarFeatures(BaseModel):
    km_driven: int
    owner: int
    mileage: float
    engine: float
    max_power: float
    seats: float
    car_age: int
    power_per_cc: float

    fuel_Diesel: bool
    fuel_LPG: bool
    fuel_Petrol: bool
    seller_type_Individual: bool
    seller_type_Trustmark_Dealer: bool
    transmission_Manual: bool

    brand_Chevrolet: bool
    brand_Daewoo: bool
    brand_Datsun: bool
    brand_Fiat: bool
    brand_Ford: bool
    brand_Honda: bool
    brand_Hyundai: bool
    brand_Kia: bool
    brand_Mahindra: bool
    brand_Maruti: bool
    brand_Nissan: bool
    brand_Opel: bool
    brand_Peugeot: bool
    brand_Renault: bool
    brand_Skoda: bool
    brand_Tata: bool
    brand_Toyota: bool
    brand_Volkswagen: bool


    

    

@app.post("/predict")
def predict_price(features: CarFeatures):
    input_array = np.array([[
        features.km_driven,
        features.owner,
        features.mileage,
        features.engine,
        features.max_power,
        features.seats,
        features.car_age,
        features.power_per_cc,
        features.fuel_Diesel,
        features.fuel_LPG,
        features.fuel_Petrol,
        features.seller_type_Individual,
        features.seller_type_Trustmark_Dealer,
        features.transmission_Manual,
        features.brand_Chevrolet,
        features.brand_Daewoo,
        features.brand_Datsun,
        features.brand_Fiat,
        features.brand_Ford,
        features.brand_Honda,
        features.brand_Hyundai,
        features.brand_Kia,
        features.brand_Mahindra,
        features.brand_Maruti,
        features.brand_Nissan,
        features.brand_Opel,
        features.brand_Peugeot,
        features.brand_Renault,
        features.brand_Skoda,
        features.brand_Tata,
        features.brand_Toyota,
        features.brand_Volkswagen
    ]])
    
    prediction = model.predict(input_array)
    return {"predicted_price": float(prediction[0])}
