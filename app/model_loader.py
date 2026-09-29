import mlflow.sklearn
import pickle


# def load_from_mlflow(model_name: str, version: str):
#     """Cargar el modelo desde el Registro de Modelos de MLflow por número de versión."""
#     
#     mlflow.set_tracking_uri("sqlite:///mlflow.db")
#     model_uri = f"models:/{model_name}/{version}"
#     print(f"Cargando modelo desde MLflow: {model_uri}")
#     # Cargar directamente como modelo de sklearn — nos proporciona predict_proba
#     return mlflow.sklearn.load_model(model_uri)


def load_from_pkl(model_path: str):
    """Cargar el modelo desde un archivo pickle local."""
    print(f"Cargando modelo desde el archivo local: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)