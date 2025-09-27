# --- main.py ---
# This is our FastAPI application file for the backend.

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd

# --- 1. Pydantic Model for Input Data Validation ---
class CustomerData(BaseModel):
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0)
    Partner: str = Field(..., example="No")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=1)
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No")
    InternetService: str = Field(..., example="Fiber optic")
    OnlineSecurity: str = Field(..., example="No")
    OnlineBackup: str = Field(..., example="No")
    DeviceProtection: str = Field(..., example="No")
    TechSupport: str = Field(..., example="No")
    StreamingTV: str = Field(..., example="No")
    StreamingMovies: str = Field(..., example="No")
    Contract: str = Field(..., example="Month-to-month")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check")
    MonthlyCharges: float = Field(..., example=70.7)
    TotalCharges: float = Field(..., example=70.7)

# --- 2. FastAPI App Initialization ---
app = FastAPI(
    title="Churn Prediction API",
    description="A production-ready API to predict customer churn using a Scikit-learn model.",
    version="1.0.0"
)

# --- 3. Load Artifacts at Startup ---
model = joblib.load('model_artifacts/model.joblib')
scaler = joblib.load('model_artifacts/scaler.joblib')
TRAINING_COLUMNS = joblib.load('model_artifacts/training_columns.joblib')

# --- 4. API Endpoints ---
@app.get("/", tags=["Status"])
def read_root():
    """A simple endpoint to check if the API is running."""
    return {"status": "ok", "message": "API is running successfully."}


@app.post("/predict", tags=["Prediction"])
def predict_churn(customer_data: CustomerData):
    """Receives customer data, preprocesses it, and returns a churn prediction."""
    input_df = pd.DataFrame([customer_data.model_dump()])
    
    # Preprocessing
    input_df_encoded = pd.get_dummies(input_df)
    input_df_aligned = input_df_encoded.reindex(columns=TRAINING_COLUMNS, fill_value=0)
    input_df_scaled = scaler.transform(input_df_aligned)
    
    # Prediction
    prediction = model.predict(input_df_scaled)
    probability = model.predict_proba(input_df_scaled)
    
    return {
        "prediction": "Yes" if prediction[0] == 1 else "No",
        "probability_of_churn": probability[0][1]
    }