# src/explicar_modelo.py
import os
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import pickle

# Limitar hilos: más estable en contenedores/Airflow
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# Sklearn
from sklearn.inspection import permutation_importance

# TreeInterpreter (añadir a requirements: treeinterpreter>=0.2.3)
try:
    from treeinterpreter import treeinterpreter as ti
    _HAS_TREEINTERPRETER = True
except Exception:
    _HAS_TREEINTERPRETER = False


def explicar_modelo(
    cleaned_path=None,
    model_path=None,
    holdout_indices_path=None,
    max_rows=300,
    top_k_features=None,   # no se usa aquí; lo dejamos por compatibilidad con tu DAG
    check_additivity=False # idem
):
    # ===== Rutas y salidas =====
    AIRFLOW_HOME = Path(os.getenv("AIRFLOW_HOME", "/opt/airflow"))
    OUT_DIR = AIRFLOW_HOME / "explicability"             # <— fuera de data
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    perm_out_path  = str(OUT_DIR / f"permutation_importance_{run_id}.csv")
    local_out_path = str(OUT_DIR / f"local_explanations_treeinterpreter_{run_id}.csv")

    if cleaned_path is None:
        cleaned_path = str(AIRFLOW_HOME / "data" / "dataset_limpio.csv")
    if model_path is None:
        model_path = str(AIRFLOW_HOME / "modelos" / "random_forest_model.pkl")

    # ===== 1) Cargar datos y modelo =====
    df = pd.read_csv(cleaned_path)

    target_candidates = ["churn", "Churn", "target"]
    target_col = next((c for c in target_candidates if c in df.columns), None)
    if target_col is not None:
        y = df[target_col]
        X = df.drop(columns=[target_col])
    else:
        y = None
        X = df.copy()

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    # ===== 2) Selección de filas (holdout o muestra) =====
    if holdout_indices_path and os.path.exists(holdout_indices_path):
        try:
            idx_df = pd.read_csv(holdout_indices_path)
            col_idx = "idx" if "idx" in idx_df.columns else idx_df.columns[0]
            idx = [i for i in idx_df[col_idx].tolist() if i in X.index]
        except Exception:
            idx = X.sample(min(max_rows, len(X)), random_state=42).index.tolist()
    else:
        idx = X.sample(min(max_rows, len(X)), random_state=42).index.tolist()

    if not idx:
        idx = X.sample(min(max_rows, len(X)), random_state=42).index.tolist()

    X_explain = X.loc[idx]
    y_explain = y.loc[idx] if y is not None else None

    if len(X_explain) > max_rows:
        X_explain = X_explain.sample(max_rows, random_state=42)
        if y_explain is not None:
            y_explain = y_explain.loc[X_explain.index]

    print(f"[explicar_modelo] Filas a explicar: {len(X_explain)} | Features: {X_explain.shape[1]}")

    # ===== 3) GLOBAL — Permutation Importance =====
    try:
        if y_explain is None:
            raise ValueError("No se encontró la columna objetivo; Permutation Importance requiere y.")
        print("[explicar_modelo] Calculando Permutation Importance…")
        r = permutation_importance(
            model, X_explain, y_explain.values,
            n_repeats=10, random_state=42, n_jobs=1  # n_jobs=1: estable en Airflow
        )
        imp_df = (pd.DataFrame({
            "feature": X_explain.columns,
            "importance_mean": r.importances_mean,
            "importance_std": r.importances_std
        }).sort_values("importance_mean", ascending=False))
        imp_df.to_csv(perm_out_path, index=False)
        print(f"[explicar_modelo] Guardado ranking global en: {perm_out_path}")
    except Exception as e:
        print(f"[explicar_modelo][WARN] Permutation Importance falló: {e}")
        perm_out_path = None

    # ===== 4) LOCAL — TreeInterpreter (clasificación: seleccionar clase/canal) =====
    print(f"[explicar_modelo] _HAS_TREEINTERPRETER={_HAS_TREEINTERPRETER}, "
          f"model_type={type(model)}, has_estimators={hasattr(model,'estimators_')}")

    local_out_final = None

    if _HAS_TREEINTERPRETER and hasattr(model, "estimators_"):
        try:
            print("[explicar_modelo] Calculando explicaciones locales (TreeInterpreter)…")

            # Asegura numérico
            X_num = X_explain.apply(pd.to_numeric, errors="coerce").astype("float64")
            X_arr = np.nan_to_num(X_num.to_numpy(copy=True), copy=False)

            preds, bias, contribs = ti.predict(model, X_arr)
            # preds   : (n_muestras, n_clases)
            # bias    : (n_muestras, n_clases)
            # contribs: (n_muestras, n_features, n_clases)

            # Clase positiva (1) si existe; si no, la de mayor prob media
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    cls_idx = classes.index(1)
                else:
                    cls_idx = int(np.argmax(np.mean(preds, axis=0)))
                cls_label = classes[cls_idx]
            else:
                cls_idx = 0
                cls_label = None

            preds_1c    = preds[:, cls_idx]        # (n_muestras,)
            bias_1c     = bias[:, cls_idx]         # (n_muestras,)
            contribs_1c = contribs[:, :, cls_idx]  # (n_muestras, n_features)

            # DataFrame “estilo internet”: contribs por feature + bias + pred + clase
            df_contribs = pd.DataFrame(contribs_1c, columns=X_num.columns)
            df_contribs.insert(0, "row_index", X_num.index.astype(int).values)
            df_contribs["bias"]  = bias_1c
            df_contribs["pred"]  = preds_1c
            df_contribs["class"] = cls_label

            df_contribs.to_csv(local_out_path, index=False)
            local_out_final = local_out_path
            print(f"[explicar_modelo] Guardadas explicaciones locales en: {local_out_final}")

        except Exception as e:
            print(f"[explicar_modelo][WARN] TreeInterpreter falló: {e}")
            local_out_final = None
    else:
        print(f"[explicar_modelo] Omito locales: _HAS_TREEINTERPRETER={_HAS_TREEINTERPRETER}, "
              f"has_estimators={hasattr(model,'estimators_')}")
        local_out_final = None

    # ===== 5) Resultado =====
    result = {
        "permutation_importance_path": perm_out_path,
        "local_explanations_path": local_out_final,
        "rows_explained": int(len(X_explain)),
        "features_used": int(X_explain.shape[1]),
    }
    print("[explicar_modelo] Resultado:", result)
    
    
    # ===== 6) Guardar en PostgreSQL =====
    insert_json(
        table_name="model_explainability",
        json_data=result,
        extra_fields={"run_id": run_id}
    )

    return result
    
    return result
