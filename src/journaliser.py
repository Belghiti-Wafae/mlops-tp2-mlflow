
import json
import subprocess
from pathlib import Path

import mlflow
import pandas as pd
import yaml

# Configuration MLflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("tp3-dvc")

# Charger les paramètres
with open("params.yaml", encoding="utf-8") as f:
    params = yaml.safe_load(f)

# Lire l'empreinte réelle des données suivies par DVC
with open("data/transactions.csv.dvc", encoding="utf-8") as f:
    ticket = yaml.safe_load(f)["outs"][0]

# Charger les métriques produites par l'évaluation
with open("reports/metrics.json", encoding="utf-8") as f:
    metrics = json.load(f)

# Informations de reproductibilité
commit = subprocess.check_output(
    ["git", "rev-parse", "--short", "HEAD"],
    text=True
).strip()

n = len(pd.read_csv("data/transactions.csv"))

# Créer une exécution MLflow
with mlflow.start_run(
    run_name=f"donnees-{ticket['md5'][:8]}"
):
    mlflow.log_params(params["entrainement"])
    mlflow.log_metrics({
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
    })

    mlflow.set_tags({
        "commit": commit,
        "donnees_md5": ticket["md5"],
        "donnees_lignes": str(n),
    })

    mlflow.log_artifact("dvc.lock")

print(
    "Run MLflow créé : données",
    ticket["md5"][:8],
    "-",
    n,
    "lignes"
)
