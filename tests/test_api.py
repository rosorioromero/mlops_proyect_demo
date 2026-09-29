# tests/test_api.py
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


# ── Verificación de estado ─────────────────────────────────────────
def test_health_check():
    """Verifica que la API esté activo y devuelva un estado saludable con código HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "saludable"


# ── Predicción local — entrada válida ─────────────────────────────
def test_predict_local_valid_input():
    """Hace una peticion POST al endpoint /predict/local con datos validos y verifica que devuelva una respuesta valida."""
    payload = {
        "Age": 35,
        "BusinessTravel": 1,
        "DailyRate": 800,
        "Department": 1,
        "DistanceFromHome": 10,
        "Education": 3,
        "EducationField": 0,
        "EnvironmentSatisfaction": 2,
        "Gender": 1,
        "HourlyRate": 60,
        "JobInvolvement": 3,
        "JobLevel": 2,
        "JobRole": 1,
        "JobSatisfaction": 2,
        "MaritalStatus": 1,
        "MonthlyIncome": 5000,
        "MonthlyRate": 15000,
        "NumCompaniesWorked": 3,
        "OverTime": 1,
        "PercentSalaryHike": 13,
        "PerformanceRating": 3,
        "RelationshipSatisfaction": 2,
        "StockOptionLevel": 1,
        "TotalWorkingYears": 8,
        "TrainingTimesLastYear": 2,
        "WorkLifeBalance": 2,
        "YearsAtCompany": 3,
        "YearsInCurrentRole": 2,
        "YearsSinceLastPromotion": 1,
        "YearsWithCurrManager": 2
    }
    response = client.post("/predict/local", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert "attrition_risk" in result
    assert "probability" in result
    assert "model_source" in result
    assert result["attrition_risk"] in ["Alto", "Bajo"]
    assert 0.0 <= result["probability"] <= 1.0


# ── Predicción local — entrada inválida activa error 422 ───────────
def test_predict_local_invalid_input():
    """Hace una peticion POST al endpoint /predict/local con datos invalidos y verifica que devuelva un error 422."""
    payload = {"Age": "not a number", "MonthlyIncome": 5000}
    response = client.post("/predict/local", json=payload)
    assert response.status_code == 422


# ── Predicción local — campos faltantes activa error 422 ───────────
def test_predict_local_missing_fields():
    """Hace una peticion POST al endpoint /predict/local con campos faltantes y verifica que devuelva un error 422."""
    payload = {"Age": 35}
    response = client.post("/predict/local", json=payload)
    assert response.status_code == 422