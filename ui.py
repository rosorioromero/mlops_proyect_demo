import streamlit as st
import requests

# ── Configuración de la página ─────────────────────────────────────
st.set_page_config(page_title="AttritionGuard", page_icon="👥")
st.title("👥 Attrition Guard")
st.caption("Predice el riesgo de deserción de empleados")

# ── Configuración de la API ────────────────────────────────────────
API_URL = "http://localhost:8001"

MODEL_CONFIG = {
    "local":  f"{API_URL}/predict/local",
    "mlflow": f"{API_URL}/predict/mlflow",
}

# ── Selector de modelo ─────────────────────────────────────────────
st.subheader("Fuente del modelo")
model_choice = st.radio(
    "MODELO",
    options=["local", "mlflow"],
    horizontal=True
)
st.code(f"MODELO   = {model_choice}\nendpoint = {MODEL_CONFIG[model_choice]}", language="bash")

st.divider()

# ── Formulario de entrada ──────────────────────────────────────────
st.subheader("Detalles del empleado")

col1, col2, col3 = st.columns(3)

with col1:
    Age                     = st.number_input("Edad",                                  min_value=18,  max_value=65,    value=35)
    MonthlyIncome           = st.number_input("Ingreso mensual",                       min_value=1,   max_value=100000, value=5000)
    TotalWorkingYears       = st.number_input("Años totales trabajados",               min_value=0,   max_value=40,    value=8)
    YearsAtCompany          = st.number_input("Años en la empresa",                    min_value=0,   max_value=40,    value=3)
    YearsInCurrentRole      = st.number_input("Años en el cargo actual",               min_value=0,   max_value=20,    value=2)
    YearsSinceLastPromotion = st.number_input("Años desde el último ascenso",          min_value=0,   max_value=20,    value=1)
    YearsWithCurrManager    = st.number_input("Años con el gerente actual",            min_value=0,   max_value=20,    value=2)
    NumCompaniesWorked      = st.number_input("Nº de empresas trabajadas",             min_value=0,   max_value=10,    value=3)
    TrainingTimesLastYear   = st.number_input("Capacitaciones el año pasado",          min_value=0,   max_value=10,    value=2)
    PercentSalaryHike       = st.number_input("Porcentaje de aumento salarial",        min_value=0,   max_value=100,   value=13)

with col2:
    JobSatisfaction          = st.slider("Satisfacción laboral",                       min_value=1, max_value=4, value=2)
    EnvironmentSatisfaction  = st.slider("Satisfacción con el entorno",                min_value=1, max_value=4, value=2)
    RelationshipSatisfaction = st.slider("Satisfacción en relaciones interpersonales", min_value=1, max_value=4, value=2)
    WorkLifeBalance          = st.slider("Equilibrio trabajo-vida",                    min_value=1, max_value=4, value=2)
    JobInvolvement           = st.slider("Involucramiento en el trabajo",              min_value=1, max_value=4, value=3)
    PerformanceRating        = st.slider("Calificación de desempeño",                  min_value=1, max_value=4, value=3)
    Education                = st.slider("Nivel educativo",                            min_value=1, max_value=5, value=3)
    JobLevel                 = st.slider("Nivel de puesto",                            min_value=1, max_value=5, value=2)
    StockOptionLevel         = st.slider("Nivel de opciones sobre acciones",           min_value=0, max_value=3, value=1)

with col3:
    OverTime       = st.selectbox("Horas extra",              options=[0, 1],    index=1,  format_func=lambda x: "Sí" if x == 1 else "No")
    Gender         = st.selectbox("Género",                   options=[0, 1],    index=1,  format_func=lambda x: "Masculino" if x == 1 else "Femenino")
    MaritalStatus  = st.selectbox("Estado civil",             options=[0, 1, 2], index=0,  format_func=lambda x: ["Soltero(a)", "Casado(a)", "Divorciado(a)"][x])
    BusinessTravel = st.selectbox("Viajes de negocios",       options=[0, 1, 2], index=1,  format_func=lambda x: ["Sin viajes", "Viaja raramente", "Viaja frecuentemente"][x])
    Department     = st.selectbox("Departamento",             options=[0, 1, 2], index=1,  format_func=lambda x: ["RRHH", "I+D", "Ventas"][x])
    EducationField = st.selectbox("Campo de estudio",         options=[0, 1, 2, 3, 4, 5], index=1, format_func=lambda x: ["RRHH", "Ciencias de la vida", "Marketing", "Medicina", "Otro", "Técnico"][x])
    JobRole        = st.selectbox("Puesto de trabajo",        options=list(range(9)), index=2, format_func=lambda x: ["Rep. de salud", "RRHH", "Técnico de laboratorio", "Gerente", "Dir. de fabricación", "Dir. de investigación", "Científico de investigación", "Ejecutivo de ventas", "Rep. de ventas"][x])
    DailyRate      = st.number_input("Tarifa diaria",         min_value=1, max_value=2000,  value=800)
    HourlyRate     = st.number_input("Tarifa por hora",       min_value=1, max_value=200,   value=60)
    MonthlyRate    = st.number_input("Tarifa mensual",        min_value=1, max_value=30000, value=15000)
    DistanceFromHome = st.number_input("Distancia desde casa", min_value=0, max_value=100, value=10)

# ── Botón de predicción ────────────────────────────────────────────
st.divider()
if st.button("Predecir riesgo de deserción", type="primary", use_container_width=True):

    payload = {
        "Age":                     Age,
        "BusinessTravel":          BusinessTravel,
        "DailyRate":               DailyRate,
        "Department":              Department,
        "DistanceFromHome":        DistanceFromHome,
        "Education":               Education,
        "EducationField":          EducationField,
        "EnvironmentSatisfaction": EnvironmentSatisfaction,
        "Gender":                  Gender,
        "HourlyRate":              HourlyRate,
        "JobInvolvement":          JobInvolvement,
        "JobLevel":                JobLevel,
        "JobRole":                 JobRole,
        "JobSatisfaction":         JobSatisfaction,
        "MaritalStatus":           MaritalStatus,
        "MonthlyIncome":           MonthlyIncome,
        "MonthlyRate":             MonthlyRate,
        "NumCompaniesWorked":      NumCompaniesWorked,
        "OverTime":                OverTime,
        "PercentSalaryHike":       PercentSalaryHike,
        "PerformanceRating":       PerformanceRating,
        "RelationshipSatisfaction":RelationshipSatisfaction,
        "StockOptionLevel":        StockOptionLevel,
        "TotalWorkingYears":       TotalWorkingYears,
        "TrainingTimesLastYear":   TrainingTimesLastYear,
        "WorkLifeBalance":         WorkLifeBalance,
        "YearsAtCompany":          YearsAtCompany,
        "YearsInCurrentRole":      YearsInCurrentRole,
        "YearsSinceLastPromotion": YearsSinceLastPromotion,
        "YearsWithCurrManager":    YearsWithCurrManager,
    }

    try:
        endpoint_url = MODEL_CONFIG[model_choice]
        response     = requests.post(endpoint_url, json=payload)

        if response.status_code == 200:
            result = response.json()
            risk   = result["attrition_risk"]
            prob   = result["probability"]
            source = result["model_source"]

            st.divider()
            if risk in ["High", "Alto"]:
                st.error(f"⚠️  Riesgo de deserción: **{risk}**")
            else:
                st.success(f"✅  Riesgo de deserción: **{risk}**")

            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Probabilidad", f"{prob:.2%}")
            with col_b:
                st.metric("Fuente del modelo", source)

        else:
            st.error(f"Error de la API {response.status_code}: {response.json().get('detail')}")

    except requests.exceptions.ConnectionError:
        st.error("❌  No se puede conectar con la API. Asegúrate de que FastAPI esté ejecutándose en el puerto 8001.")