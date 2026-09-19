from pathlib import Path
import os

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

from src.advanced import multi_horizon_forecast, validate_dataset
from src.energy_optimizer import load_dataset

app = FastAPI(title="GridWise AI API", version="2.0.0", description="Energy forecasting and decision-support service")
DATA_PATH = Path("data/energydata_complete.csv")
API_KEY = os.getenv("GRIDWISE_API_KEY")


def require_api_key(x_api_key: str | None = Header(default=None)):
    """Require a shared key when GRIDWISE_API_KEY is configured."""
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


class ForecastRequest(BaseModel):
    horizon: int = Field(default=24, ge=1, le=168)


@app.get("/health")
def health():
    return {"status": "ok", "dataset_available": DATA_PATH.exists()}


@app.get("/quality", dependencies=[Depends(require_api_key)])
def quality():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset unavailable")
    return validate_dataset(load_dataset(DATA_PATH)).__dict__


@app.post("/forecast", dependencies=[Depends(require_api_key)])
def forecast(request: ForecastRequest):
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Dataset unavailable; run download_data.py")
    output = multi_horizon_forecast(load_dataset(DATA_PATH), request.horizon)
    output["timestamp"] = output["timestamp"].astype(str)
    return output.to_dict(orient="records")
