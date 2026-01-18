import os
import psycopg2
import json

def get_connection():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "postgres"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        dbname=os.getenv("POSTGRES_DB", "airflow"),
        user=os.getenv("POSTGRES_USER", "airflow"),
        password=os.getenv("POSTGRES_PASSWORD", "airflow")
    )

def insert_json(table_name, json_data, extra_fields=None):
    conn = get_connection()
    cur = conn.cursor()
    payload = json.dumps(json_data)
    
    if extra_fields:
        columns = ', '.join(extra_fields.keys()) + ', data'
        values = [*extra_fields.values(), payload]
        placeholders = ', '.join(['%s'] * len(values))
    else:
        columns = 'data'
        values = [payload]
        placeholders = '%s'
    
    query = f"INSERT INTO {table_name} ({columns}) VALUES ({placeholders})"
    cur.execute(query, values)
    conn.commit()
    cur.close()
    conn.close()
