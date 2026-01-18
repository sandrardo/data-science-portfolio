import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.inspection import PartialDependenceDisplay

def generar_pdp(**context):
    ti = context["ti"]

    # === Recuperar rutas desde XCom ===
    entrenar_result = ti.xcom_pull(task_ids="entrenar_modelo")
    explicar_result = ti.xcom_pull(task_ids="explicar_modelo")

    model_path = entrenar_result["model_path"]
    x_test_path = entrenar_result["x_test_path"]
    y_test_path = entrenar_result["y_test_path"]
    ranking_path = explicar_result["permutation_importance_path"]
    contrib_path = explicar_result["local_explanations_path"]

    top_k = int(os.getenv("TOP_K_FEATURES", "5"))
    explic_dir = Path(os.getenv("EXPLICABILITY_DIR", "/opt/airflow/explicability"))
    explic_dir.mkdir(parents=True, exist_ok=True)

    # === Cargar modelo y datos ===
    with open(model_path, "rb") as f:
        model = pickle.load(f)

    X_test = pd.read_csv(x_test_path)

    # === Leer ranking global ===
    ranking_df = pd.read_csv(ranking_path)
    top_global = ranking_df.sort_values("importance_mean", ascending=False).head(top_k)

    # === Leer contribuciones locales ===
    contrib_df = pd.read_csv(contrib_path)
    contrib_only = contrib_df.drop(columns=["row_index", "bias", "pred", "class"], errors="ignore")
    mean_contrib = contrib_only.abs().mean().sort_values(ascending=False).head(top_k)

    # === Comparar rankings ===
    print("\n📊 Top K Features — Comparación Global vs Local:")
    for i in range(top_k):
        g_feat = top_global.iloc[i]["feature"]
        g_imp = top_global.iloc[i]["importance_mean"]
        l_imp = mean_contrib.get(g_feat, 0)
        print(f"🔹 {g_feat:<20} | Global: {g_imp:.4f} | Local: {l_imp:.4f}")

    # === Guardar rankings en CSV ===
    top_global.to_csv(explic_dir / "top_features_global.csv", index=False)

    # Convertir mean_contrib (Series) a DataFrame antes de guardar
    mean_contrib_df = mean_contrib.reset_index()
    mean_contrib_df.columns = ["feature", "local_mean_contrib"]
    mean_contrib_df.to_csv(explic_dir / "top_features_local.csv", index=False)

    print(f"\n📁 Rankings guardados en:\n- Global: {explic_dir / 'top_features_global.csv'}\n- Local: {explic_dir / 'top_features_local.csv'}")

    # === Seleccionar features para PDP ===
    selected_features = top_global["feature"].tolist()
    print(f"\n🧮 Generando PDPs para: {selected_features}")

    # === Graficar PDPs ===
    fig, ax = plt.subplots(figsize=(12, 6 * top_k))
    display = PartialDependenceDisplay.from_estimator(
        model,
        X_test,
        features=selected_features,
        ax=ax
    )

    pdp_path = explic_dir / "partial_dependence_plots.png"
    fig.tight_layout()
    fig.savefig(pdp_path)
    print(f"\n✅ PDPs guardados en: {pdp_path}")


    # === Guardar en PostgreSQL ===
    run_id = entrenar_result.get("run_id", "unknown")

    pdp_result = {
        "top_k": top_k,
        "selected_features": selected_features,
        "global_ranking_path": str(top_global_path),
        "local_ranking_path": str(top_local_path),
        "pdp_image_path": str(pdp_path)
    }

    insert_json("model_pdp", pdp_result, extra_fields={"run_id": run_id})

    return str(pdp_path)

