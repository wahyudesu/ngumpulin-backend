from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.models import Variable
from airflow.utils.log.logging_mixin import LoggingMixin
from mlflow_plugin import MLflowModelOperator, MLflowExperimentOperator
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import pickle
import json
import os
import glob
import logging
from supabase import create_client, Client
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import optuna

# Setup logging
logger = logging.getLogger(__name__)

# MLflow settings
MLFLOW_TRACKING_URI = "http://mlflow:5000"
EXPERIMENT_NAME = "document-processing"
MODEL_NAME = "document-clustering"

default_args = {
    'owner': 'ngumpulin',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
    'params': {
        'n_trials': Variable.get('n_trials', 20),
        'silhouette_threshold': Variable.get('silhouette_threshold', 0.5),
        'model_dir': Variable.get('model_dir', '/tmp/models'),
    }
}

dag = DAG(
    'ml_pipeline',
    default_args=default_args,
    description='ML Pipeline for document processing',
    # schedule_interval=timedelta(days=1),   # production interval
    schedule_interval=timedelta(minutes=3),  # testing interval
    catchup=False,
)

# Create MLflow experiment using plugin operator
create_experiment = MLflowExperimentOperator(
    task_id='create_experiment',
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    experiment_name=EXPERIMENT_NAME,
    dag=dag
)

def extract_data(**context):
    """Extract data from Supabase and save as CSV."""
    try:
        with mlflow.start_run(run_name="data_extraction"):
            supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
            table_name = "documents"
            response = supabase.table(table_name).select("*").execute()
            df = pd.DataFrame(response.data)
            
            # Log data statistics
            mlflow.log_metric("total_documents", len(df))
            mlflow.log_metric("total_sentences", df['sentences'].sum())
            
            out_path = "/tmp/extracted_data.csv"
            df.to_csv(out_path, index=False)
            mlflow.log_artifact(out_path, "raw_data")
            
            logger.info(f"Data extracted successfully to {out_path}")
            return out_path
    except Exception as e:
        logger.error(f"Error in extract_data: {str(e)}")
        raise

def validate_data(**context):
    """Validate data before processing."""
    try:
        with mlflow.start_run(run_name="data_validation"):
            in_path = context['ti'].xcom_pull(task_ids='extract_data')
            df = pd.read_csv(in_path)
            
            required_columns = ['sentences', 'page', 'timing', 'plagiarism']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check for null values
            null_counts = df[required_columns].isnull().sum()
            if null_counts.any():
                logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
                mlflow.log_metric("null_values_count", null_counts.sum())
                
            # Check data types
            if not all(df['sentences'].apply(lambda x: isinstance(x, (int, float)))):
                raise ValueError("'sentences' column must be numeric")
                
            # Log validation metrics
            mlflow.log_metric("validation_passed", 1)
            logger.info("Data validation completed successfully")
            return True
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

def preprocess_data(**context):
    """Preprocess the extracted data and save as CSV."""
    try:
        with mlflow.start_run(run_name="data_preprocessing"):
            in_path = context['ti'].xcom_pull(task_ids='extract_data')
            df = pd.read_csv(in_path)
            
            # Date conversion and feature engineering
            df['deadline'] = pd.to_datetime(df['deadline'])
            df['uploadedDate'] = pd.to_datetime(df['uploadedDate'])
            df['timing'] = (df['deadline'] - df['uploadedDate']).dt.total_seconds() / 3600
            df['timing'] = df['timing'].astype(int)
            
            # Plagiarism value
            import ast
            def extract_plagiarism(row):
                try:
                    if isinstance(row, str):
                        row = ast.literal_eval(row)
                    return round(max([v for item in row for v in item.values()]) * 100, 2) if row else 0
                except Exception:
                    return 0
            df['plagiarism'] = df['plagiarism'].apply(extract_plagiarism)
            
            # Select features
            data = df[['sentences', 'page', 'timing', 'plagiarism']]
            
            # Scaling
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(data)
            X_scaled = pd.DataFrame(X_scaled, columns=data.columns)
            
            # Weighting
            weights = np.array([0.5, 0.5, 1.5, 4.5])
            X_weight = X_scaled * weights
            X_weight = pd.DataFrame(X_weight, columns=data.columns)
            
            # Save scaler and weighted data
            scaler_path = "/tmp/scaler.pkl"
            weighted_path = "/tmp/preprocessed_data.csv"
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            X_weight.to_csv(weighted_path, index=False)
            
            # Log preprocessing artifacts
            mlflow.log_artifact(scaler_path, "preprocessing")
            mlflow.log_artifact(weighted_path, "preprocessing")
            
            # Log preprocessing metrics
            mlflow.log_metric("preprocessed_samples", len(X_weight))
            mlflow.log_param("feature_weights", weights.tolist())
            
            logger.info("Data preprocessing completed successfully")
            return weighted_path
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

def train_model(**context):
    """Run clustering, optimize with Optuna, and save model."""
    try:
        with mlflow.start_run(run_name="model_training"):
            weighted_path = context['ti'].xcom_pull(task_ids='preprocess_data')
            X_weight = pd.read_csv(weighted_path)
            
            # Define Optuna objectives
            def objective_kmeans(trial):
                n_clusters = trial.suggest_int('n_clusters', 2, 3)
                init_method = trial.suggest_categorical('init', ['k-means++', 'random'])
                n_init = trial.suggest_int('n_init', 1, 10)
                max_iter = trial.suggest_int('max_iter', 100, 1000)
                algorithm = trial.suggest_categorical('algorithm', ['auto', 'full', 'elkan'])
                random_state = trial.suggest_int('random_state', 0, 1000)
                
                model = KMeans(
                    n_clusters=n_clusters, 
                    init=init_method, 
                    n_init=n_init,
                    max_iter=max_iter,
                    algorithm=algorithm,
                    random_state=random_state
                )
                
                try:
                    labels = model.fit_predict(X_weight)
                    if len(set(labels)) <= 1:
                        return -1.0
                    score = silhouette_score(X_weight, labels)
                    return score
                except Exception:
                    return -1.0
                    
            def objective_bisecting_kmeans(trial):
                n_clusters = trial.suggest_int('n_clusters', 2, 3)
                init = trial.suggest_categorical('init', ['k-means++', 'random'])
                n_init = trial.suggest_int('n_init', 1, 10)
                max_iter = trial.suggest_int('max_iter', 100, 1000)
                random_state = trial.suggest_int('random_state', 0, 1000)
                
                model = BisectingKMeans(
                    n_clusters=n_clusters,
                    init=init,
                    n_init=n_init,
                    max_iter=max_iter,
                    random_state=random_state
                )
                
                try:
                    labels = model.fit_predict(X_weight)
                    if len(set(labels)) <= 1:
                        return -1.0
                    score = silhouette_score(X_weight, labels)
                    return score
                except Exception:
                    return -1.0
                    
            def objective_gmm(trial):
                n_components = trial.suggest_int('n_components', 2, 3)
                covariance_type = trial.suggest_categorical('covariance_type', ['full', 'tied', 'diag', 'spherical'])
                init_params = trial.suggest_categorical('init_params', ['kmeans', 'random'])
                random_state = trial.suggest_int('random_state', 0, 1000)
                
                model = GaussianMixture(
                    n_components=n_components,
                    covariance_type=covariance_type,
                    init_params=init_params,
                    random_state=random_state
                )
                
                try:
                    labels = model.fit_predict(X_weight)
                    if len(set(labels)) <= 1:
                        return -1.0
                    return silhouette_score(X_weight, labels)
                except Exception:
                    return -1.0
                    
            algorithms = {
                'KMeans': objective_kmeans,
                'BisectingKMeans': objective_bisecting_kmeans,
                'GaussianMixture': objective_gmm,
            }
            
            n_trials = context['params']['n_trials']
            results = {}
            best_params = {}
            
            for algo_name, objective in algorithms.items():
                study = optuna.create_study(direction='maximize')
                study.optimize(objective, n_trials=n_trials)
                results[algo_name] = study.best_value
                best_params[algo_name] = study.best_params
                
                # Log Optuna results
                mlflow.log_metric(f"{algo_name}_best_score", study.best_value)
                mlflow.log_params({f"{algo_name}_{k}": v for k, v in study.best_params.items()})
                
            # Model selection
            best_algo = max(results, key=lambda x: results[x])
            params = best_params[best_algo]
            
            if best_algo == 'KMeans':
                model = KMeans(**params)
            elif best_algo == 'BisectingKMeans':
                model = BisectingKMeans(**params)
            elif best_algo == 'GaussianMixture':
                model = GaussianMixture(**params)
                
            model.fit(X_weight)
            
            # Save model
            model_path = "/tmp/best_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            # Log training metrics
            mlflow.log_metric("best_algorithm", best_algo)
            mlflow.log_metric("best_silhouette_score", results[best_algo])
            
            logger.info(f"Model training completed successfully. Best algorithm: {best_algo}")
            return model_path
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

def evaluate_model(**context):
    """Evaluate model performance and log metrics to MLflow."""
    try:
        with mlflow.start_run(run_name="model_evaluation"):
            # Load the latest model from MLflow
            model = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
            weighted_path = context['ti'].xcom_pull(task_ids='preprocess_data')
            X_weight = pd.read_csv(weighted_path)
            
            # Make predictions
            labels = model.predict(X_weight)
            
            # Calculate metrics
            silhouette = silhouette_score(X_weight, labels)
            calinski = calinski_harabasz_score(X_weight, labels)
            davies = davies_bouldin_score(X_weight, labels)
            
            # Log metrics to MLflow
            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("calinski_harabasz_score", calinski)
            mlflow.log_metric("davies_bouldin_score", davies)
            
            # Log cluster statistics
            cluster_sizes = pd.Series(labels).value_counts()
            for cluster, size in cluster_sizes.items():
                mlflow.log_metric(f"cluster_{cluster}_size", size)
            
            logger.info(f"Model evaluation - Silhouette: {silhouette:.4f}, Calinski: {calinski:.4f}, Davies: {davies:.4f}")
            return True
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def monitor_model_performance(**context):
    """Monitor model performance and alert if degradation detected."""
    try:
        with mlflow.start_run(run_name="performance_monitoring"):
            # Get the latest model metrics
            client = mlflow.tracking.MlflowClient()
            latest_run = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
                filter_string="tags.mlflow.runName = 'model_evaluation'",
                max_results=1
            )[0]
            
            current_metrics = {
                'silhouette': latest_run.data.metrics['silhouette_score'],
                'calinski': latest_run.data.metrics['calinski_harabasz_score'],
                'davies': latest_run.data.metrics['davies_bouldin_score']
            }
            
            # Get previous metrics
            previous_runs = client.search_runs(
                experiment_ids=[mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id],
                filter_string="tags.mlflow.runName = 'model_evaluation'",
                max_results=2
            )
            
            if len(previous_runs) > 1:
                previous_metrics = {
                    'silhouette': previous_runs[1].data.metrics['silhouette_score'],
                    'calinski': previous_runs[1].data.metrics['calinski_harabasz_score'],
                    'davies': previous_runs[1].data.metrics['davies_bouldin_score']
                }
                
                # Check for significant degradation
                threshold = 0.1  # 10% degradation threshold
                for metric in current_metrics:
                    if current_metrics[metric] < previous_metrics[metric] * (1 - threshold):
                        logger.warning(f"Significant degradation detected in {metric}")
                        mlflow.log_metric(f"{metric}_degradation", 1)
                    else:
                        mlflow.log_metric(f"{metric}_degradation", 0)
            
            logger.info("Model performance monitoring completed")
            return True
    except Exception as e:
        logger.error(f"Error in performance monitoring: {str(e)}")
        raise

def deploy_model(**context):
    """Deploy the best model."""
    try:
        model_path = context['ti'].xcom_pull(task_ids='train_model')
        # Copy model to production location
        prod_path = "/app/models/best_model.pkl"
        os.makedirs(os.path.dirname(prod_path), exist_ok=True)
        with open(model_path, 'rb') as src, open(prod_path, 'wb') as dst:
            dst.write(src.read())
        logger.info(f"Model deployed successfully to {prod_path}")
        return True
    except Exception as e:
        logger.error(f"Error in model deployment: {str(e)}")
        raise

# Create tasks
extract_task = PythonOperator(
    task_id='extract_data',
    python_callable=extract_data,
    dag=dag,
)

validate_task = PythonOperator(
    task_id='validate_data',
    python_callable=validate_data,
    dag=dag,
)

preprocess_task = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

train_task = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)

# Register model using MLflow plugin operator
register_model = MLflowModelOperator(
    task_id='register_model',
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    model_path="{{ ti.xcom_pull(task_ids='train_model') }}",
    model_name=MODEL_NAME,
    dag=dag
)

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

monitor_task = PythonOperator(
    task_id='monitor_performance',
    python_callable=monitor_model_performance,
    dag=dag,
)

deploy_task = PythonOperator(
    task_id='deploy_model',
    python_callable=deploy_model,
    dag=dag,
)

# Set task dependencies
create_experiment >> extract_task >> validate_task >> preprocess_task >> train_task >> register_model >> evaluate_task >> monitor_task >> deploy_task