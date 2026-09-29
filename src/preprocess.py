import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os
import pickle

def preprocess(data_path: str, output_dir: str):
    """
    Carga y preprocesa el conjunto de datos de refugiados de ACNUR.

    Pasos:
    - Eliminar columnas que no aportan información
    - Codificar la columna objetivo (Refugio_Otorgado: Yes/No -> 1/0)
    - Codificar todas las columnas categóricas usando LabelEncoder
    - Dividir en conjuntos de entrenamiento y prueba

    Args:
        data_path  : ruta al archivo CSV original
        output_dir : carpeta donde se guardarán los archivos procesados
    """

    print("=" * 50)
    print("PASO 1: Cargando datos")
    print("=" * 50)
    df = pd.read_csv(data_path)
    print(f"Forma del conjunto de datos: {df.shape}")
    print(f"Columnas: {list(df.columns)}")


    # ------------------------------------------------------------------ #
    # Eliminar columnas que no aportan información útil
    # Over18        -> todos los empleados tienen 'Y', varianza cero
    # StandardHours -> todos los empleados tienen 80, varianza cero
    # EmployeeCount -> siempre es 1
    # EmployeeNumber -> solo es un ID, no una característica
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("\nPASO 2: Eliminando columnas con poca información")
    print("=" * 50)
    cols_to_drop = ["Over18", "StandardHours", "EmployeeCount", "EmployeeNumber"]

    # Solo eliminar columnas que realmente existan en el conjunto de datos
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    print(f"Eliminadas: {cols_to_drop}")
    print(f"Forma después de eliminar: {df.shape}")


    # ------------------------------------------------------------------ #
    # Codificar columna objetivo
    # Attrition (Deserción): Yes -> 1, No -> 0
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("\nPASO 3: Codificando columna objetivo (Attrition)")
    print("=" * 50)
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    print(f"Conteo de valores de Attrition:\n{df['Attrition'].value_counts()}")


    # ------------------------------------------------------------------ #
    # Codificar columnas categóricas
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("\nPASO 4: Codificando columnas categóricas")
    print("=" * 50)
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"Columnas categóricas encontradas: {categorical_cols}")

    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    print("Todas las columnas categóricas han sido codificadas.")


    # ------------------------------------------------------------------ #
    # Separar características y variable objetivo
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("\nPASO 5: Dividiendo en características (X) y variable objetivo (y)")
    print("=" * 50)
    X = df.drop(columns=["Attrition"])
    y = df["Attrition"]
    print(f"Forma de las características: {X.shape}")
    print(f"Forma de la variable objetivo: {y.shape}")


    # ------------------------------------------------------------------ #
    # División en entrenamiento / prueba
    # 80% entrenamiento, 20% prueba
    # stratify=y asegura la misma proporción de clases en ambas divisiones
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print("\nPASO 6: División entrenamiento/prueba (80/20, estratificado)")
    print("=" * 50)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    print(f"X_train: {X_train.shape} | X_test: {X_test.shape}")
    print(f"y_train: {y_train.shape} | y_test: {y_test.shape}")


    # ------------------------------------------------------------------ #
    # Guardar datos procesados
    # ------------------------------------------------------------------ #
    print("=" * 50)
    print(f"\nPASO 7: Guardando datos procesados en '{output_dir}/'")
    print("=" * 50)
    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(f"{output_dir}/X_train.csv", index=False)
    X_test.to_csv(f"{output_dir}/X_test.csv", index=False)
    y_train.to_csv(f"{output_dir}/y_train.csv", index=False)
    y_test.to_csv(f"{output_dir}/y_test.csv", index=False)

    # Guardar los codificadores para reutilizarlos durante la inferencia???
    with open(f"{output_dir}/label_encoders.pkl", "wb") as f:
        pickle.dump(label_encoders, f)

    print("Guardados: X_train.csv, X_test.csv, y_train.csv, y_test.csv")
    print("Guardado: label_encoders.pkl")
    print("\n¡Preprocesamiento completo!")

    return X_train, X_test, y_train, y_test


# ------------------------------------------------------------------ #
# Ejecutar directamente
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    preprocess(
        data_path="data/HR-Employee-Attrition.csv",
        output_dir="data/processed"
    )