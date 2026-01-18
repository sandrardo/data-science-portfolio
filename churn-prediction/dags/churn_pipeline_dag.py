from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

import sys
sys.path.insert(0, '/opt/airflow/src')

from prepare_data import prepare_data
from train_model import train_model
from evaluate_model import evaluate_model
from explain_model import explain_model
from generate_pdp import generate_pdp

default_args = {
    'start_date': datetime(2023, 1, 1),
    'retries': 0,
}

def wrapper_entrenar_modelo(**kwargs):
    resultado = train_model()
    run_id = resultado["run_id"]
    kwargs['ti'].xcom_push(key='run_id', value=run_id)

def wrapper_evaluar_modelo(**kwargs):
    run_id = kwargs['ti'].xcom_pull(key='run_id', task_ids='entrenar_modelo')
    evaluate_model(run_id=run_id)

def wrapper_preparar_datos(**kwargs):
    resultado = prepare_data(**kwargs)
    kwargs['ti'].xcom_push(key='cleaned_path', value=resultado["cleaned_path"])
    kwargs['ti'].xcom_push(key='row_count', value=resultado["row_count"])

with DAG(
    dag_id='modelo_churn',
    default_args=default_args,
    description='DAG para preparar datos, entrenar y evaluar un modelo de churn',
    schedule=None,
    catchup=False
) as dag:

    t1 = PythonOperator(
        task_id='preparar_datos',
        python_callable=wrapper_preparar_datos
    )

    t2 = PythonOperator(
        task_id='entrenar_modelo',
        python_callable=wrapper_entrenar_modelo
    )

    t3 = PythonOperator(
        task_id='evaluar_modelo',
        python_callable=wrapper_evaluar_modelo
    )

    t4 = PythonOperator(
        task_id="explicar_modelo",
        python_callable=explicar_modelo,
        op_kwargs={
            "max_rows": 5000,
            "check_additivity": False,
            "scoring": "roc_auc"
        },
    )

    t5 = PythonOperator(
        task_id="generar_pdp",
        python_callable=generar_pdp,
        op_kwargs={"top_k": 5},
    )

    # Orden de ejecución
    t1 >> t2 >> [t3, t4]
    t4 >> t5
