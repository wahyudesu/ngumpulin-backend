from prometheus_client import start_http_server, Gauge, Counter
import mlflow
import time
import os
from datetime import datetime, timedelta

# Define metrics
experiment_count = Gauge('mlflow_experiments_total', 'Total number of experiments')
run_count = Gauge('mlflow_runs_total', 'Total number of runs')
model_count = Gauge('mlflow_models_total', 'Total number of registered models')
run_duration = Gauge('mlflow_run_duration_seconds', 'Duration of runs in seconds')
run_metrics = Gauge('mlflow_run_metric', 'Run metrics', ['run_id', 'metric_name'])

def collect_metrics():
    """Collect metrics from MLflow."""
    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
        
        # Get experiments
        experiments = mlflow.search_experiments()
        experiment_count.set(len(experiments))
        
        # Get runs
        runs = mlflow.search_runs(experiment_ids=[exp.experiment_id for exp in experiments])
        run_count.set(len(runs))
        
        # Get registered models
        client = mlflow.tracking.MlflowClient()
        models = client.search_registered_models()
        model_count.set(len(models))
        
        # Get run metrics
        for run in runs:
            run_id = run.info.run_id
            metrics = client.get_run(run_id).data.metrics
            for metric_name, value in metrics.items():
                run_metrics.labels(run_id=run_id, metric_name=metric_name).set(value)
            
            # Calculate run duration
            start_time = datetime.fromtimestamp(run.info.start_time / 1000)
            end_time = datetime.fromtimestamp(run.info.end_time / 1000) if run.info.end_time else datetime.now()
            duration = (end_time - start_time).total_seconds()
            run_duration.labels(run_id=run_id).set(duration)
            
    except Exception as e:
        print(f"Error collecting metrics: {str(e)}")

if __name__ == '__main__':
    # Start Prometheus metrics server
    start_http_server(8001)
    
    # Collect metrics every 30 seconds
    while True:
        collect_metrics()
        time.sleep(30)