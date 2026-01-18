import pandas as pd
import os
import pickle
import mlflow
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

from db_utils import insert_json  # 👈 Asegúrate de que este módulo esté en tu PYTHONPATH

def evaluate_model(
    run_id,
    cleaned_path="/opt/airflow/data/dataset_limpio.csv",
    model_path="/opt/airflow/modelos/random_forest_model.pkl",
    holdout_indices_path=None):

    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("modelo_prediccion")

    df = pd.read_csv(cleaned_path)
    X = df.drop("churn", axis=1); y = df["churn"]

    if holdout_indices_path and os.path.exists(holdout_indices_path):
        idx = pd.read_csv(holdout_indices_path)["idx"].tolist()
        X_test, y_test = X.loc[idx], y.loc[idx]
    else:
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

    with open(model_path, "rb") as f:
        modelo = pickle.load(f)

    y_pred = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    resultados = pd.DataFrame({
        "churn_real": y_test,
        "churn_predicho": y_pred,
        "probabilidad_churn": y_proba
    })

    os.makedirs("/opt/airflow/results", exist_ok=True)
    output_path = "/opt/airflow/results/predicciones_churn.csv"
    resultados.to_csv(output_path, index=False)

    cm_df = pd.DataFrame(cm, columns=["Pred_No", "Pred_Sí"], index=["Real_No", "Real_Sí"])
    cm_path = "/opt/airflow/results/matriz_confusion.csv"
    cm_df.to_csv(cm_path)

    cm_img_path = "/opt/airflow/results/matriz_confusion.png"
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No", "Sí"], yticklabels=["No", "Sí"])
    plt.xlabel("Predicción")
    plt.ylabel("Real")
    plt.title("Matriz de Confusión")
    plt.savefig(cm_img_path)
    plt.close()

    print("✅ Evaluación del modelo:")
    print(f"  Accuracy: {accuracy}")
    print(f"  ROC AUC:  {roc_auc}")
    print("📊 Matriz de confusión:")
    print(cm)

    # Registrar en MLflow
    mlflow.start_run(run_id=run_id)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("precision", report["1"]["precision"])
    mlflow.log_metric("recall", report["1"]["recall"])
    mlflow.log_metric("f1_score", report["1"]["f1-score"])
    mlflow.log_artifact(cm_img_path)
    mlflow.log_artifact(output_path)
    mlflow.log_artifact(cm_path)
    mlflow.end_run()

    # Registrar en PostgreSQL
    metricas = {
        "accuracy": float(accuracy),
        "roc_auc": float(roc_auc),
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"],
        "confusion_matrix": cm.tolist(),
        "artifacts": {
            "predictions_csv": output_path,
            "confusion_matrix_csv": cm_path,
            "confusion_matrix_img": cm_img_path
        }
    }

    insert_json("model_metrics", metricas, extra_fields={"run_id": run_id})

    return metricas

