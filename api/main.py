import json
import logging
from pathlib import Path
from uuid import uuid4

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger(__name__)

API_VERSION = "1.0.0"
MODEL_VERSION = "week1-v1"

APP_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = APP_DIR / "ml_platform" / "training" / "churn" / "models"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
FEATURES_PATH = MODELS_DIR / "feature_names.json"

FIELD_TO_FEATURE = {
    "call_failure": "Call Failure",
    "complains": "Complains",
    "subscription_length": "Subscription Length",
    "charge_amount": "Charge Amount",
    "seconds_of_use": "Seconds of Use",
    "frequency_of_use": "Frequency of use",
    "frequency_of_sms": "Frequency of SMS",
    "distinct_called_numbers": "Distinct Called Numbers",
    "age_group": "Age Group",
    "tariff_plan": "Tariff Plan",
    "status": "Status",
    "age": "Age",
    "customer_value": "Customer Value",
}

for path in (MODEL_PATH, FEATURES_PATH):
    if not path.exists():
        raise RuntimeError(
            f"Required artifact missing: {path}. "
            "Run ml_platform/training/churn/train.py first."
        )

model = joblib.load(MODEL_PATH)
feature_names = json.loads(FEATURES_PATH.read_text())
logger.info(
    "model_loaded path=%s model_version=%s",
    MODEL_PATH,
    MODEL_VERSION,
)

app = FastAPI(
    title="Ishango Telecom Churn API",
    version=API_VERSION,
    description="Week 1 production API for telecom churn prediction.",
)


class ChurnRequest(BaseModel):
    call_failure: int = Field(ge=0, description="Number of failed calls")
    complains: int = Field(ge=0, le=1, description="Customer complaint flag (0/1)")
    subscription_length: int = Field(ge=0, description="Months subscribed")
    charge_amount: float = Field(ge=0, description="Charge amount band")
    seconds_of_use: float = Field(ge=0, description="Total seconds of calls")
    frequency_of_use: float = Field(ge=0, description="Total number of calls")
    frequency_of_sms: float = Field(ge=0, description="Total number of SMS")
    distinct_called_numbers: float = Field(ge=0, description="Distinct numbers called")
    age_group: int = Field(ge=0, description="Age group code")
    tariff_plan: int = Field(ge=0, description="Tariff plan code")
    status: int = Field(ge=0, description="Account status code")
    age: int = Field(ge=0, description="Customer age")
    customer_value: float = Field(ge=0, description="Calculated customer value")


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float = Field(ge=0, le=1)
    model_version: str
    request_id: str


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "telecom-churn-api",
        "model_version": MODEL_VERSION,
        "api_version": API_VERSION,
    }


@app.get("/version")
def version() -> dict[str, str]:
    return {
        "api_version": API_VERSION,
        "model_version": MODEL_VERSION,
    }


@app.post("/predictions", response_model=PredictionResponse)
def predict(payload: ChurnRequest) -> PredictionResponse:
    request_id = str(uuid4())

    values = {
        FIELD_TO_FEATURE[field]: value for field, value in payload.model_dump().items()
    }
    row = pd.DataFrame(
        [[values[name] for name in feature_names]],
        columns=feature_names,
    )

    try:
        prediction = int(model.predict(row)[0])
        churn_probability = float(model.predict_proba(row)[0][1])
    except Exception as exc:
        logger.exception("prediction_failed request_id=%s", request_id)
        raise HTTPException(
            status_code=500,
            detail="Prediction service failed to process the request.",
        ) from exc

    logger.info(
        "prediction_completed request_id=%s prediction=%s model_version=%s",
        request_id,
        prediction,
        MODEL_VERSION,
    )
    return PredictionResponse(
        prediction=prediction,
        churn_probability=round(churn_probability, 4),
        model_version=MODEL_VERSION,
        request_id=request_id,
    )
