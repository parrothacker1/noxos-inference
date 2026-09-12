import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Request

from service.clamav_client import ClamdUnavailable, scan_bytes

MODEL_PATH = Path(__file__).resolve().parent / "model_network.joblib"
API_KEY = os.environ.get("NOXOS_INFERENCE_API_KEY", "")
MODE = os.environ.get("NOXOS_INFERENCE_MODE", "both")

_model_bundle = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

app = FastAPI(title="noxos-inference")


def require_auth(authorization: str | None = Header(default=None)):
    if not API_KEY:
        return
    if authorization != f"Bearer {API_KEY}":
        raise HTTPException(status_code=401, detail="invalid or missing API key")


def require_mode(mode: str):
    if MODE not in (mode, "both"):
        raise HTTPException(status_code=404, detail=f"this deployment does not serve /{mode}")


def encode_request(payload: dict[str, Any]) -> pd.DataFrame:
    bundle = _model_bundle
    row = {}
    for col in bundle["feature_columns"]:
        if col in bundle["categorical_columns"]:
            value = str(payload.get(col, bundle["categorical_defaults"][col]))
            categories = bundle["categories"][col]
            row[col] = categories.index(value) if value in categories else -1
        else:
            row[col] = payload.get(col, bundle["numeric_defaults"][col])
    return pd.DataFrame([row], columns=bundle["feature_columns"])


def bucket_verdict(attack_probability: float) -> str:
    if attack_probability < 0.4:
        return "allow"
    if attack_probability > 0.6:
        return "block"
    return "uncertain"


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model_bundle is not None, "mode": MODE}


@app.post("/analyze/network")
def analyze_network(payload: dict[str, Any] = Body(default={}), _auth: None = Depends(require_auth)):
    require_mode("network")
    if _model_bundle is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    features = encode_request(payload)
    attack_probability = float(_model_bundle["model"].predict_proba(features)[0][1])
    return {
        "verdict": bucket_verdict(attack_probability),
        "safety_score": round(1.0 - attack_probability, 4),
        "reasoning": None,
    }


@app.post("/analyze/file")
async def analyze_file(request: Request, _auth: None = Depends(require_auth)):
    require_mode("file")
    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="empty request body, expected raw file bytes")

    try:
        result, signature = scan_bytes(body)
    except ClamdUnavailable as e:
        return {"verdict": "uncertain", "safety_score": None, "reasoning": str(e)}

    if result == "OK":
        return {"verdict": "allow", "safety_score": 1.0, "reasoning": None}
    if result == "FOUND":
        return {"verdict": "block", "safety_score": 0.0, "reasoning": f"clamd: {signature}"}
    return {"verdict": "uncertain", "safety_score": None, "reasoning": f"clamd error: {signature}"}
