
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Create the FastAPI application
app = FastAPI(title="House Price Prediction API")

# Load the trained model
model = joblib.load("models/house_model.joblib")


# Define the input data structure
class HouseDetails(BaseModel):
    area_sqft: int
    bedrooms: int
    bathrooms: int
    house_age: int


# Create the prediction endpoint
@app.post("/predict")
def predict_price(house: HouseDetails):

    input_data = [[
        house.area_sqft,
        house.bedrooms,
        house.bathrooms,
        house.house_age
    ]]

    prediction = model.predict(input_data)

    return {
        "predicted_price": round(float(prediction[0]), 2)
    }
