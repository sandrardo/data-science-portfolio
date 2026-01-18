import pandas as pd
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn
from db_utils import insert_json

def entrenar_modelo():
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("modelo_prediccion")

    os.makedirs("/opt/airflow/modelos", exist_ok=True)
    os.makedirs("/opt/airflow/data", exist_ok=True)

    df = pd.read_csv("/opt/airflow/data/dataset_limpio.csv")
    X = df.drop("churn", axis=1)
    y = df["churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = RandomForestClassifier(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)

    model_path = "/opt/airflow/modelos/random_forest_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(modelo, f)

    imp_path = "/opt/airflow/modelos/feature_importance.csv"
    importancia = pd.DataFrame({
        "feature": X.columns,
        "importancia": modelo.feature_importances_
    }).sort_values("importancia", ascending=False)
    importancia.to_csv(imp_path, index=False)

    holdout_path = "/opt/airflow/data/holdout_indices.csv"
    pd.DataFrame({"idx": X_test.index}).to_csv(holdout_path, index=False)

    x_test_path = "/opt/airflow/data/X_test.csv"
    y_test_path = "/opt/airflow/data/y_test.csv"
    X_test.to_csv(x_test_path, index=False)
    y_test.to_csv(y_test_path, index=False)

    with mlflow.start_run() as run:
        run_id = run.info.run_id

        mlflow.log_param("modelo", "RandomForest")
        mlflow.log_param("n_estimators", 100)

        mlflow.sklearn.log_model(modelo, "modelo_rf")
        mlflow.log_artifact(model_path)
        mlflow.log_artifact(imp_path)
        mlflow.log_artifact(holdout_path)
        mlflow.log_artifact(x_test_path)
        mlflow.log_artifact(y_test_path)

    # Guardar resumen en PostgreSQL
    resumen = {
        "model_type": "RandomForest",
        "parameters": {
            "n_estimators": 100,
            "random_state": 42
        },
        "artifacts": {
            "model_path": model_path,
            "feature_importance_path": imp_path,
            "holdout_indices_path": holdout_path,
            "x_test_path": x_test_path,
            "y_test_path": y_test_path
        },
        "top_features": importancia.head(10).to_dict(orient="records")
    }

    insert_json("training_log", resumen, extra_fields={"run_id": run_id})

    print(f"[entrenar_modelo] Modelo guardado en: {model_path}")
    print(f"[entrenar_modelo] train/test: {X_train.shape} / {X_test.shape}")
    print("[entrenar_modelo] Top 10 features:")
    print(importancia.head(10).to_string(index=False))
    print(f"[entrenar_modelo] Run ID: {run_id}")

    return {
        "run_id": run_id,
        "model_path": model_path,
        "feature_importance_path": imp_path,
        "holdout_indices_path": holdout_path,
        "x_test_path": x_test_path,
        "y_test_path": y_test_path
    }
