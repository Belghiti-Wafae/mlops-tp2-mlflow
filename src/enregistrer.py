import mlflow


mlflow.set_tracking_uri(
    'http://127.0.0.1:5000'
)


exp = mlflow.get_experiment_by_name(
    'tp2-registre'
)


runs = mlflow.search_runs(
    [exp.experiment_id],
    order_by=['metrics.f1 DESC']
)


for _, r in runs.head(3).iterrows():

    mv = mlflow.register_model(
        f"runs:/{r['run_id']}/model",
        'fraude'
    )

    print(
        'version',
        mv.version,
        '- F1',
        round(
            r['metrics.f1'],
            4
        )
    )