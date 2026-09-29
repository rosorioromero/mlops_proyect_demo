import pandas as pd
import numpy as np

def simulate_drift(input_path: str, output_path: str, random_state: int = 42):
    """
    Simula la desviación de datos (data drift) en el conjunto de datos de AttritionGuard.
    
    Historia: 6 meses después del despliegue, la empresa pasó por
    una reestructuración. Los empleados están trabajando más horas extra,
    ganando menos y están menos satisfechos.
    """
    np.random.seed(random_state)
    
    df = pd.read_csv(input_path)
    drifted = df.copy()
    n = len(drifted)

    print(f"Forma del conjunto de datos original: {df.shape}")
    print("\nSimulando data drift...")

    # ── Desviación 1: OverTime (Horas Extra) ─────────────────────────
    # Originalmente ~30% horas extra. Ahora ~60% — la reestructuración
    # significa menos personas haciendo más trabajo
    original_overtime = drifted["OverTime"].mean()
    drifted["OverTime"] = np.random.choice([0, 1], size=n, p=[0.4, 0.6])
    print(f"Promedio de OverTime      : {original_overtime:.2f} → {drifted['OverTime'].mean():.2f}")

    # ── Desviación 2: JobSatisfaction (Satisfacción Laboral) ─────────
    # Originalmente distribuido de 1 a 4. Ahora sesgado a valores más bajos — la moral cayó
    original_js = drifted["JobSatisfaction"].mean()
    drifted["JobSatisfaction"] = np.random.choice([1, 2, 3, 4], size=n, p=[0.4, 0.35, 0.15, 0.1])
    print(f"Promedio de JobSatisfaction: {original_js:.2f} → {drifted['JobSatisfaction'].mean():.2f}")

    # ── Desviación 3: MonthlyIncome (Ingreso Mensual) ───────────────
    # Desplaza la distribución de ingresos hacia abajo — congelación de contrataciones, rangos más bajos
    original_income = drifted["MonthlyIncome"].mean()
    drifted["MonthlyIncome"] = (drifted["MonthlyIncome"] * 0.75).astype(int)
    print(f"Promedio de MonthlyIncome : {original_income:.0f} → {drifted['MonthlyIncome'].mean():.0f}")

    # ── Desviación 4: WorkLifeBalance (Balance Trabajo-Vida) ───────
    # Sesgado hacia abajo — más personas reportan un deficiente balance trabajo-vida
    original_wlb = drifted["WorkLifeBalance"].mean()
    drifted["WorkLifeBalance"] = np.random.choice([1, 2, 3, 4], size=n, p=[0.35, 0.35, 0.2, 0.1])
    print(f"Promedio de WorkLifeBalance: {original_wlb:.2f} → {drifted['WorkLifeBalance'].mean():.2f}")

    drifted.to_csv(output_path, index=False)
    print(f"\nConjunto de datos con drift guardado: {output_path}")
    print(f"Forma: {drifted.shape}")

    return drifted


if __name__ == "__main__":
    simulate_drift(
        input_path  = "data/processed/X_train.csv",
        output_path = "data/processed/X_drifted.csv"
    )