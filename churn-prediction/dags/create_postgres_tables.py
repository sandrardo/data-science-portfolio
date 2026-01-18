# dags/create_postgres_tables.py
from airflow import DAG
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from datetime import datetime

with DAG(
    dag_id="create_postgres_tables",
    start_date=datetime(2023, 1, 1),
    schedule=None,          # solo se lanza a mano
    catchup=False,
    tags=["setup"],
) as dag:

    create_tables = SQLExecuteQueryOperator(
        task_id="create_tables",
        conn_id="postgres_default",                  # <- ojo: conn_id (no postgres_conn_id)
        sql="sql/create_tables.sql",    # ruta dentro del contenedor
    )
