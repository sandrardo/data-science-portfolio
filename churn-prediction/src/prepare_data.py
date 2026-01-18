# src/preparar_datos.py

import pandas as pd
import os
from collections import Counter
from airflow.providers.postgres.hooks.postgres import PostgresHook
import io # Necesario para manejar el dataframe en memoria

def preparar_datos(**context):
    os.makedirs("data", exist_ok=True)

    src_path = "/opt/airflow/data/dataset.csv"
    output_path = "/opt/airflow/data/dataset_limpio.csv"

    df = pd.read_csv(src_path, sep=";", decimal=",")

    # ... (toda tu lógica de limpieza de datos sigue igual)
    columnas_a_eliminar = [
        "Customer_ID", "ethnic", "kid0_2", "kid3_5", "kid6_10", "kid11_15", "kid16_17",
        "creditcd", "truck", "rv", "numbcars", "infobase", "HHstatin", "ownrent"
    ]
    df = df.drop(columns=columnas_a_eliminar)
    df["new_cell"] = df["new_cell"].fillna("U")
    df["prizm_social_one"] = df["prizm_social_one"].fillna("U")
    df["dualband"] = df["dualband"].fillna("U")
    df["marital"] = df["marital"].fillna("U")
    df["hnd_webcap"] = df["hnd_webcap"].fillna("UNKW")
    categorias_modo = ["crclscod", "asl_flag", "area", "refurb_new", "dwlltype", "dwllsize"]
    for col in categorias_modo:
        df[col] = df[col].fillna(df[col].mode()[0])
    columnas_numericas_con_nulos = [
        "lor", "income", "adults", "avg6mou", "avg6rev", "avg6qty",
        "change_rev", "change_mou", "hnd_price", "datovr_Mean", "roam_Mean",
        "mou_Mean", "totmrc_Mean", "da_Mean", "ovrmou_Mean", "ovrrev_Mean",
        "rev_Mean", "vceovr_Mean", "models", "phones", "eqpdays", "forgntvl"
    ]
    for col in columnas_numericas_con_nulos:
        df[col] = df[col].fillna(df[col].median())
    columnas_categoricas = [
        "new_cell", "crclscod", "asl_flag", "prizm_social_one", "area",
        "dualband", "refurb_new", "hnd_webcap", "dwlltype", "marital", "dwllsize"
    ]
    df = pd.get_dummies(df, columns=columnas_categoricas, drop_first=True)
    
    df.to_csv(output_path, index=False)

    # --- INICIO DEL CÓDIGO CORREGIDO ---
    print("[preparar_datos] Conectando a PostgreSQL para guardar los datos...")
    hook = PostgresHook(postgres_conn_id='postgres_default')
    conn = hook.get_conn()
    cursor = conn.cursor()
    
    table_name = 'transformed_data'

    try:
        print(f"[preparar_datos] Preparando la carga masiva para la tabla '{table_name}'...")
        
        # 1. Borramos la tabla si ya existe y la creamos de nuevo.
        #    Esto asegura que la estructura coincida con el dataframe.
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        
        # Generamos el comando CREATE TABLE a partir de los tipos de datos del dataframe
        # Reemplazamos los tipos de pandas por tipos de SQL
        dtype_mapping = {
            'int64': 'INTEGER',
            'float64': 'REAL',
            'bool': 'BOOLEAN',
            'object': 'TEXT'
        }
        # Los nombres de columna en SQL no deben contener caracteres especiales
        df.columns = [c.replace(' ', '_').replace('.', '') for c in df.columns]
        sql_dtypes = {col: dtype_mapping.get(str(df[col].dtype), 'TEXT') for col in df.columns}
        create_table_sql = f"CREATE TABLE {table_name} ({', '.join([f'\"{col}\" {dtype}' for col, dtype in sql_dtypes.items()])});"
        
        cursor.execute(create_table_sql)
        print(f"[preparar_datos] Tabla '{table_name}' creada.")

        # 2. Preparamos los datos en un buffer de memoria con formato CSV.
        buffer = io.StringIO()
        df.to_csv(buffer, index=False, header=False, sep='\t') # Usamos tabulador como separador
        buffer.seek(0) # Rebobinamos el buffer al principio

        # 3. Usamos copy_expert para una carga masiva y eficiente.
        #    Es el método más rápido para insertar datos en PostgreSQL.
        print("[preparar_datos] Realizando carga masiva con COPY...")
        cursor.copy_expert(f"COPY {table_name} FROM STDIN WITH (FORMAT CSV, DELIMITER E'\\t')", buffer)
        
        conn.commit()
        print(f"[preparar_datos] Datos guardados exitosamente en la tabla '{table_name}'.")

    except Exception as e:
        print(f"Error al guardar en la base de datos: {e}")
        conn.rollback()
        raise
    finally:
        # 4. Cerramos cursor y conexión.
        cursor.close()
        conn.close()
        print("[preparar_datos] Conexión a la base de datos cerrada.")
    # --- FIN DEL CÓDIGO CORREGIDO ---

    print(f"[preparar_datos] Origen: {src_path}")
    print(f"[preparar_datos] Salida: {output_path}")
    print(f"[preparar_datos] Filas x Columnas: {df.shape}")
    if "churn" in df.columns:
        dist = Counter(df["churn"])
        print(f"[preparar_datos] Distribución churn: {dict(dist)}")

    return {"cleaned_path": output_path, "row_count": int(len(df))}
