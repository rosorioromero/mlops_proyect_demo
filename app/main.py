# http://127.0.0.1:8000/docs

from fastapi import FastAPI, HTTPException
from app.schemas import EmployeeInput, PredictionOutput
from app.model_loader import load_from_pkl  #, load_from_mlflow agregas si usas MLflow
import pandas as pd

app = FastAPI(
    title="AttritionGuard API",
    description="Predice el riesgo de deserción de empleados",
    version="1.0.0"
)

# ── Cargar ambos modelos al iniciar ────────────────────────────────
# mlflow_model = load_from_mlflow("AttritionGuard", version="1")
local_model  = load_from_pkl("models/model.pkl")


# ── Verificación de estado ─────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "saludable"}


# # ── Predicción usando el modelo de MLflow ──────────────────────────
# @app.post("/predict/mlflow", response_model=PredictionOutput)
# def predict_mlflow(employee: EmployeeInput):
#     """Predicción usando el modelo cargado desde el registro de MLflow."""
#     try:
#         input_df     = pd.DataFrame([employee.model_dump()])
#         probability  = float(mlflow_model.predict_proba(input_df)[0][1])
#         return PredictionOutput(
#             attrition_risk = "Alto" if probability >= 0.5 else "Bajo",
#             probability    = round(probability, 4),
#             model_source   = "Registro de MLflow — versión 1"
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# ── Predicción usando el modelo pkl local ─────────────────────────
@app.post("/predict/local", response_model=PredictionOutput)
def predict_local(employee: EmployeeInput):
    """Predicción usando el modelo cargado desde el archivo pickle local."""
    try:
        input_df     = pd.DataFrame([employee.model_dump()])
        probability  = float(local_model.predict_proba(input_df)[0][1])
        return PredictionOutput(
            attrition_risk = "Alto" if probability >= 0.5 else "Bajo",
            probability    = round(probability, 4),
            model_source   = "Archivo local — models/model.pkl"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))