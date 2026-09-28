import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    ConfusionMatrixDisplay
)


# ------------------------------------------------------------------ #
# Auxiliar: cargar datos procesados
# ------------------------------------------------------------------ #
def load_data(processed_dir: str):
    X_train = pd.read_csv(f"{processed_dir}/X_train.csv")
    X_test  = pd.read_csv(f"{processed_dir}/X_test.csv")
    y_train = pd.read_csv(f"{processed_dir}/y_train.csv").squeeze()
    y_test  = pd.read_csv(f"{processed_dir}/y_test.csv").squeeze()
    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
# Auxiliar: evaluar un modelo entrenado y retornar diccionario de métricas
# ------------------------------------------------------------------ #
def evaluate(model, X_test, y_test) -> dict:
    preds      = model.predict(X_test)
    preds_prob = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy" : round(accuracy_score(y_test, preds), 4),
        "f1_score" : round(f1_score(y_test, preds), 4),
        "precision": round(precision_score(y_test, preds), 4),
        "recall"   : round(recall_score(y_test, preds), 4),
        "roc_auc"  : round(roc_auc_score(y_test, preds_prob), 4),
    }


# ------------------------------------------------------------------ #
# Auxiliar: guardar y registrar la matriz de confusión como artefacto
# ------------------------------------------------------------------ #
def log_confusion_matrix(model, X_test, y_test, run_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_estimator(
        model, X_test, y_test,
        display_labels=["Sin Deserción", "Deserción"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title(f"Matriz de Confusión — {run_name}")

    os.makedirs("artifacts", exist_ok=True)
    path = f"artifacts/confusion_matrix_{run_name}.png"
    plt.savefig(path, bbox_inches="tight")
    plt.close()

    mlflow.log_artifact(path)
    print(f"  Matriz de confusión guardada y registrada: {path}")


# ------------------------------------------------------------------ #
# EJECUCIÓN 1 — Regresión Logística
# ------------------------------------------------------------------ #
def run_logistic_regression(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("EJECUCIÓN 1: Regresión Logística")
    print("=" * 50)

    # Hiperparámetros
    C        = 1.0
    max_iter = 1000
    solver   = "lbfgs"

    with mlflow.start_run(run_name="logistic-regression"):

        # La regresión logística necesita características escaladas para converger correctamente
        # Usamos un Pipeline: StandardScaler -> LogisticRegression
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(C=C, max_iter=max_iter, solver=solver))
        ])
        model.fit(X_train, y_train)

        # --- Evaluar ---
        metrics = evaluate(model, X_test, y_test)

        # --- Registrar parámetros ---
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", C)
        mlflow.log_param("max_iter", max_iter)
        mlflow.log_param("solver", solver)

        # --- Registrar métricas ---
        print("MÉTRICAS---")
        for metric_name, metric_value in metrics.items():
            print(f"{metric_name} -> {metric_value}")
            mlflow.log_metric(metric_name, metric_value)

        # --- Registrar modelo ---
        mlflow.sklearn.log_model(model, name="model")

        # --- Registrar matriz de confusión ---
        log_confusion_matrix(model, X_test, y_test, "logistic-regression")

        # --- Imprimir resultados ---
        print(f"  Parámetros : C={C}, max_iter={max_iter}, solver={solver}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Ejecución completada.")


# ------------------------------------------------------------------ #
# EJECUCIÓN 2 — Random Forest (línea base)
# ------------------------------------------------------------------ #
def run_random_forest_baseline(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("EJECUCIÓN 2: Random Forest (línea base)")
    print("=" * 50)

    # Hiperparámetros
    n_estimators = 100
    max_depth    = 5
    random_state = 42

    with mlflow.start_run(run_name="random-forest-baseline"):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("random_state", random_state)

        print("MÉTRICAS---")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name} -> {metric_value}")
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        log_confusion_matrix(model, X_test, y_test, "random-forest-baseline")

        print(f"  Parámetros : n_estimators={n_estimators}, max_depth={max_depth}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Ejecución completada.")


# ------------------------------------------------------------------ #
# EJECUCIÓN 3 — XGBoost
# ------------------------------------------------------------------ #
def run_xgboost(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("EJECUCIÓN 3: XGBoost")
    print("=" * 50)

    # Hiperparámetros
    n_estimators    = 100
    max_depth       = 4
    learning_rate   = 0.1
    random_state    = 42

    with mlflow.start_run(run_name="xgboost"):

        model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            eval_metric="logloss",
            verbosity=0
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("random_state", random_state)

        print("MÉTRICAS---")
        for metric_name, metric_value in metrics.items():
            print(f"  {metric_name} -> {metric_value}")
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["xgboost.core.Booster", "xgboost.sklearn.XGBClassifier"])
        log_confusion_matrix(model, X_test, y_test, "xgboost")

        print(f"  Parámetros : n_estimators={n_estimators}, max_depth={max_depth}, lr={learning_rate}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Ejecución completada.")


# ------------------------------------------------------------------ #
# EJECUCIÓN 4 — Random Forest (optimizado)
# ------------------------------------------------------------------ #
def run_random_forest_tuned(X_train, X_test, y_train, y_test):
    print("\n" + "=" * 50)
    print("EJECUCIÓN 4: Random Forest (optimizado)")
    print("=" * 50)

    # Hiperparámetros optimizados
    n_estimators      = 200
    max_depth         = 10
    min_samples_split = 5
    min_samples_leaf  = 2
    random_state      = 42

    with mlflow.start_run(run_name="random-forest-tuned"):

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        metrics = evaluate(model, X_test, y_test)

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)
        mlflow.log_param("random_state", random_state)

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        mlflow.sklearn.log_model(model, name="model", skops_trusted_types=["sklearn.tree._tree.Tree"])
        log_confusion_matrix(model, X_test, y_test, "random-forest-tuned")

        print(f"  Parámetros : n_estimators={n_estimators}, max_depth={max_depth}, "
              f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        for k, v in metrics.items():
            print(f"  {k:<12}: {v}")

    print("Ejecución completada.")


# ------------------------------------------------------------------ #
# REGISTRAR EL MEJOR MODELO
# ------------------------------------------------------------------ #
def register_best_model(experiment_name: str, model_name: str):
    """
    Encuentra la mejor ejecución por puntuación ROC AUC y la registra
    en el Registro de Modelos de MLflow.
    """
    print("\n" + "=" * 50)
    print("REGISTRANDO EL MEJOR MODELO")
    print("=" * 50)

    # Buscar todas las ejecuciones en el experimento, ordenadas por roc_auc descendente
    runs = mlflow.search_runs(
        experiment_names=[experiment_name],
        order_by=["metrics.roc_auc DESC"]
    )

    best_run = runs.iloc[0]
    best_run_id   = best_run["run_id"]
    best_run_name = best_run["tags.mlflow.runName"]
    best_roc_auc  = best_run["metrics.roc_auc"]
    best_accuracy = best_run["metrics.accuracy"]

    print(f"  Mejor ejecución : {best_run_name}")
    print(f"  ID de ejecución  : {best_run_id}")
    print(f"  ROC AUC          : {best_roc_auc}")
    print(f"  Exactitud        : {best_accuracy}")

    # Registrar modelo
    model_uri = f"runs:/{best_run_id}/model"
    registered = mlflow.register_model(
        model_uri=model_uri,
        name=model_name
    )

    print(f"\n  Modelo '{model_name}' registrado.")
    print(f"  Versión          : {registered.version}")
    print(f"\n  Abre la interfaz gráfica de MLflow para cambiar este modelo a 'Staging' o 'Production'.")
    print(f"  Comando          : mlflow ui")
    print(f"  URL              : http://localhost:5000")

    # Guardar modelo localmente
    best_model = mlflow.sklearn.load_model(model_uri)
    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(best_model, f)

    print(f"  Modelo local actualizado exitosamente en 'models/model.pkl'")

# ------------------------------------------------------------------ #
# PRINCIPAL
# ------------------------------------------------------------------ #
if __name__ == "__main__":

    PROCESSED_DIR   = "data/processed"
    EXPERIMENT_NAME = "attrition-prediction"
    MODEL_NAME      = "AttritionGuard"

    # --- Cargar datos ---
    print("Cargando datos procesados...")
    X_train, X_test, y_train, y_test = load_data(PROCESSED_DIR)
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")

    # --- Configurar experimento de MLflow ---
    # Crea el experimento si aún no existe
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"\nExperimento de MLflow: '{EXPERIMENT_NAME}'")
    print("Todas las ejecuciones se registrarán aquí.\n")

    # --- Ejecutar los cuatro modelos ---
    run_logistic_regression(X_train, X_test, y_train, y_test)
    run_random_forest_baseline(X_train, X_test, y_train, y_test)
    run_xgboost(X_train, X_test, y_train, y_test)
    run_random_forest_tuned(X_train, X_test, y_train, y_test)

    # --- Registrar el mejor modelo ---
    register_best_model(EXPERIMENT_NAME, MODEL_NAME)

    print("\n" + "=" * 50)
    print("TODO LISTO")
    print("=" * 50)
    print(f"  4 ejecuciones registradas en el experimento de MLflow: '{EXPERIMENT_NAME}'")
    print(f"  El mejor modelo fue registrado como: '{MODEL_NAME}'")
    print(f"\n  Inicia la interfaz de MLflow para explorar los resultados:")
    print(f"  $ mlflow ui")
    print(f"  Luego abre: http://localhost:5000")