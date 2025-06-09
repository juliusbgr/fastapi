from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(
    title="Iris Prediction API",
    description="Predict the species of an Iris flower using a trained ML model.",
    version="1.0.0"
)

# Load trained model (loaded once when app starts)
model = joblib.load("iris.mdl")

class IrisRequest(BaseModel):
    """
    Input schema for iris flower features.
    """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

@app.post("/predict", summary="Predict Iris Species", response_description="Predicted species of the Iris flower")
def predict(data: IrisRequest):
    """
    Predicts the species of an Iris flower based on sepal and petal measurements.

    Args:
        data (IrisRequest): A JSON body with four float fields.

    Returns:
        dict: A JSON response with the predicted species.
    """
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"prediction": prediction}

# Optional: Keep the original hello endpoint
class NameRequest(BaseModel):
    name: str

@app.post("/hello", summary="Greet the user", response_description="A greeting message")
def hello(data: NameRequest):
    return {"message": f"Hello {data.name}"}
