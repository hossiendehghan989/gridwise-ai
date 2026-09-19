from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.advanced import multi_horizon_forecast, validate_dataset
from src.energy_optimizer import load_dataset

app = FastAPI(title="GridWise AI API", version="2.0.0", description="Energy forecasting and decision-support service")
DATA_PATH = Path("data/energydata_complete.csv")


class ForecastRequest(BaseModel):
    horizon: int = Field(default=24, ge=1, le=168)


@app.get("/health")
def health():
    return {"status": "ok", "dataset_available": DATA_PATH.exists()}


@app.get("/quality")
def quality():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset unavailable")
    return validate_dataset(load_dataset(DATA_PATH)).__dict__


@app.post("/forecast")
def forecast(request: ForecastRequest):
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset unavailable; run download_data.py")
    output = multi_horizon_forecast(load_dataset(DATA_PATH), request.horizon)
    output["timestamp"] = output["timestamp"].astype(str)
    return output.to_dict(orient="records")
