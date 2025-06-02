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
from mlflow.tracking import MlflowClient
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

# Initialize MLflow experiment
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
else:
    experiment_id = experiment.experiment_id
mlflow.set_experiment(experiment_id=experiment_id)

# Ensure artifact directory exists
artifact_root = "/mlflow/artifacts"
if not os.path.exists(artifact_root):
    os.makedirs(artifact_root, exist_ok=True)
    logger.info(f"✓ Created artifact directory: {artifact_root}")
else:
    logger.info(f"✓ Artifact directory exists: {artifact_root}")

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
        'model_dir': Variable.get('model_dir', '/mlflow/artifacts/models'),
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
        with mlflow.start_run(run_name="data_extraction", experiment_id=experiment_id, nested=True):
            supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
            table_name = "documents"
            response = supabase.table(table_name).select("*").execute()
            df = pd.DataFrame(response.data)
            
            # Convert sentences to numeric if it's not already
            df['sentences'] = pd.to_numeric(df['sentences'], errors='coerce')
            df['page'] = pd.to_numeric(df['page'], errors='coerce')
            
            # Calculate metrics
            total_docs = int(len(df))
            total_sentences = int(df['sentences'].sum())
            avg_sentences = float(df['sentences'].mean())
            avg_pages = float(df['page'].mean())
            
            # Log metrics
            mlflow.log_metric("total_documents", total_docs)
            mlflow.log_metric("total_sentences", total_sentences)
            mlflow.log_metric("avg_sentences", avg_sentences)
            mlflow.log_metric("avg_pages", avg_pages)
            
            out_path = "/mlflow/artifacts/extracted_data.csv"
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
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
        with mlflow.start_run(run_name="data_validation", experiment_id=experiment_id, nested=True):
            in_path = context['ti'].xcom_pull(task_ids='extract_data')
            df = pd.read_csv(in_path)
            
            required_columns = ['sentences', 'page', 'deadline', 'uploadedDate', 'plagiarism']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
                
            # Check for null values
            null_counts = df[required_columns].isnull().sum()
            total_nulls = int(null_counts.sum())
            
            if null_counts.any():
                logger.warning(f"Found null values: {null_counts[null_counts > 0]}")
                mlflow.log_metric("null_values_count", total_nulls)
                
            # Check data types
            if not all(df['sentences'].apply(lambda x: isinstance(x, (int, float)))):
                raise ValueError("'sentences' column must be numeric")
                
            # Log validation metrics
            mlflow.log_metric("validation_passed", 1.0)
            logger.info("Data validation completed successfully")
            return True
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

def preprocess_data(**context):
    """Preprocess the extracted data and save as CSV."""
    try:
        with mlflow.start_run(run_name="data_preprocessing", experiment_id=experiment_id, nested=True):
            in_path = context['ti'].xcom_pull(task_ids='extract_data')
            df = pd.read_csv(in_path)
            
            # Date conversion and feature engineering
            df['deadline'] = pd.to_datetime(df['deadline'])
            df['uploadedDate'] = pd.to_datetime(df['uploadedDate'])
            
            # Calculate timing with proper handling of NA and inf values
            df['timing'] = (df['deadline'] - df['uploadedDate']).dt.total_seconds() / 3600
            # Replace inf with NA
            df['timing'] = df['timing'].replace([np.inf, -np.inf], np.nan)
            
            # Handle NaN values in all numeric columns
            numeric_columns = ['sentences', 'page', 'timing']
            for col in numeric_columns:
                if col in df.columns:
                    # Fill NaN with median for each column
                    median_value = float(df[col].median())
                    df[col] = df[col].fillna(median_value)
                    # Log the median value used for imputation
                    mlflow.log_metric(f"{col}_median_imputation", median_value)
            
            # Convert timing to integer after handling NaN
            df['timing'] = df['timing'].astype(int)
            
            # Plagiarism value extraction with better error handling
            def extract_plagiarism(row):
                try:
                    # Handle empty string or None
                    if pd.isna(row) or row == '[]' or row == '':
                        return 0.0
                    
                    # Handle string representation of list/dict
                    if isinstance(row, str):
                        # Clean the string if needed
                        row = row.strip()
                        if row.startswith('[') and row.endswith(']'):
                            row = row[1:-1]  # Remove brackets
                        if row.startswith('{') and row.endswith('}'):
                            row = row[1:-1]  # Remove braces
                        
                        # Parse the string into a list of dictionaries
                        try:
                            # Handle multiple dictionaries
                            if ',' in row:
                                items = row.split('}, {')
                                items = [item.strip('{}') for item in items]
                                max_value = 0.0
                                for item in items:
                                    # Parse each dictionary
                                    pairs = item.split(',')
                                    for pair in pairs:
                                        if ':' in pair:
                                            key, value = pair.split(':')
                                            value = float(value.strip())
                                            max_value = max(max_value, value)
                                return round(max_value * 100, 2)
                            else:
                                # Handle single dictionary
                                pairs = row.split(',')
                                max_value = 0.0
                                for pair in pairs:
                                    if ':' in pair:
                                        key, value = pair.split(':')
                                        value = float(value.strip())
                                        max_value = max(max_value, value)
                                return round(max_value * 100, 2)
                        except Exception as e:
                            logger.warning(f"Error parsing plagiarism value: {str(e)}")
                            return 0.0
                    
                    # Handle list/dict directly
                    if isinstance(row, (list, dict)):
                        if isinstance(row, list):
                            values = [v for item in row for v in item.values()]
                        else:
                            values = list(row.values())
                        return round(max(values) * 100, 2) if values else 0.0
                    
                    return 0.0
                except Exception as e:
                    logger.warning(f"Error in extract_plagiarism: {str(e)}")
                    return 0.0
            
            # Apply plagiarism extraction and handle any remaining NaN
            df['plagiarism'] = df['plagiarism'].apply(extract_plagiarism)
            df['plagiarism'] = df['plagiarism'].fillna(0.0)
            
            # Select features
            data = df[['sentences', 'page', 'timing', 'plagiarism']]
            
            # Verify no NaN values remain
            if data.isna().any().any():
                raise ValueError("NaN values still present after preprocessing")
            
            # Scaling
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(data)
            X_scaled = pd.DataFrame(X_scaled, columns=data.columns)
            
            # Weighting
            weights = np.array([0.5, 0.5, 1.5, 4.5])  # Using the best weights from notebook
            X_weight = X_scaled * weights
            X_weight = pd.DataFrame(X_weight, columns=data.columns)
            
            # Save scaler and weighted data to shared volume
            scaler_path = "/mlflow/artifacts/scaler.pkl"
            weighted_path = "/mlflow/artifacts/preprocessed_data.csv"
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            
            with open(scaler_path, 'wb') as f:
                pickle.dump(scaler, f)
            X_weight.to_csv(weighted_path, index=False)
            
            # Log preprocessing artifacts
            mlflow.log_artifact(scaler_path, "preprocessing")
            mlflow.log_artifact(weighted_path, "preprocessing")
            
            # Log preprocessing metrics
            mlflow.log_metric("preprocessed_samples", int(len(X_weight)))
            mlflow.log_param("feature_weights", weights.tolist())
            
            # Log feature statistics
            for col in data.columns:
                mlflow.log_metric(f"{col}_mean", float(data[col].mean()))
                mlflow.log_metric(f"{col}_std", float(data[col].std()))
                mlflow.log_metric(f"{col}_min", float(data[col].min()))
                mlflow.log_metric(f"{col}_max", float(data[col].max()))
            
            logger.info("Data preprocessing completed successfully")
            return weighted_path
    except Exception as e:
        logger.error(f"Error in preprocess_data: {str(e)}")
        raise

def train_model(**context):
    """Run clustering, optimize with Optuna, and save model."""
    try:
        with mlflow.start_run(run_name="model_training", experiment_id=experiment_id, nested=True):
            weighted_path = context['ti'].xcom_pull(task_ids='preprocess_data')
            X_weight = pd.read_csv(weighted_path)
            
            # Get all previous model runs
            client = MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="tags.mlflow.runName = 'model_evaluation'",
                max_results=10  # Get last 10 runs
            )
            
            # Calculate weighted scores for each run
            best_score = -float('inf')
            best_run = None
            
            for run in runs:
                metrics = run.data.metrics
                
                # Check if all required metrics exist
                required_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score', 'training_data_size']
                if not all(metric in metrics for metric in required_metrics):
                    continue
                    
                # Get training data size
                training_data_size = float(metrics.get('training_data_size', 0))
                
                # Calculate clustering metrics score
                weights = {
                    'silhouette': 0.5,  # Higher weight for silhouette as it's most important for clustering
                    'calinski': 0.3,
                    'davies': 0.2
                }
                
                # Calculate base score (higher is better for silhouette and calinski, lower is better for davies)
                try:
                    base_score = (
                        weights['silhouette'] * float(metrics['silhouette_score']) +
                        weights['calinski'] * float(metrics['calinski_harabasz_score']) +
                        weights['davies'] * (1 - float(metrics['davies_bouldin_score']))  # Invert davies score
                    )
                except ValueError:
                    logger.warning(f"Skipping run {run.info.run_id} due to non-numeric metrics.")
                    continue
                
                # Calculate data size factor (normalize to 0-1 range)
                data_size_factor = min(training_data_size / 1000.0, 1.0)
                
                # Calculate final weighted score
                final_score = base_score * (0.7 + 0.3 * data_size_factor)
                
                if final_score > best_score:
                    best_score = final_score
                    best_run = run
            
            # Define Optuna objectives
            def objective_kmeans(trial):
                n_clusters = trial.suggest_int('n_clusters', 2, 3)
                init_method = trial.suggest_categorical('init', ['k-means++', 'random'])
                n_init = trial.suggest_int('n_init', 1, 10)
                max_iter = trial.suggest_int('max_iter', 100, 1000)
                algorithm = trial.suggest_categorical('algorithm', ['lloyd', 'elkan'])
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
                    return float(score)
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
                    return float(score)
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
                    return float(silhouette_score(X_weight, labels))
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
                results[algo_name] = float(study.best_value)
                best_params[algo_name] = study.best_params
                
                # Log Optuna results
                mlflow.log_metric(f"{algo_name}_best_score", float(study.best_value))
                for param_name, param_value in study.best_params.items():
                    if isinstance(param_value, (int, float)):
                        mlflow.log_metric(f"{algo_name}_{param_name}", float(param_value))
                    else:
                        mlflow.log_param(f"{algo_name}_{param_name}", param_value)
                
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
            
            # Calculate metrics for current model
            labels = model.predict(X_weight)
            current_metrics = {
                'silhouette_score': float(silhouette_score(X_weight, labels)),
                'calinski_harabasz_score': float(calinski_harabasz_score(X_weight, labels)),
                'davies_bouldin_score': float(davies_bouldin_score(X_weight, labels)),
                'training_data_size': float(len(X_weight))
            }
            
            # Calculate current model's score
            current_base_score = (
                weights['silhouette'] * current_metrics['silhouette_score'] +
                weights['calinski'] * current_metrics['calinski_harabasz_score'] +
                weights['davies'] * (1 - current_metrics['davies_bouldin_score'])
            )
            
            current_data_size_factor = min(current_metrics['training_data_size'] / 1000.0, 1.0)
            current_final_score = current_base_score * (0.7 + 0.3 * current_data_size_factor)
            
            # Compare with best previous model
            if best_run is not None and best_score > current_final_score:
                logger.info(f"Previous model {best_run.info.run_id} is better than current model")
                # Load and use the previous model
                model = mlflow.sklearn.load_model(f"runs:/{best_run.info.run_id}/model")
                mlflow.log_param("using_previous_model", True)
                mlflow.log_param("previous_model_run_id", best_run.info.run_id)
                mlflow.log_metric("previous_model_score", best_score)
            else:
                logger.info("Current model is better than all previous models")
                mlflow.log_param("using_previous_model", False)
                mlflow.log_metric("current_model_score", current_final_score)
            
            # Log model parameters
            mlflow.log_params(params)
            
            # Log the model as an artifact
            mlflow.sklearn.log_model(model, "model")
            
            # Log training metrics
            mlflow.log_param("best_algorithm", best_algo)
            mlflow.log_metric("best_silhouette_score", float(results[best_algo]))
            
            # Log all metrics
            for metric_name, value in current_metrics.items():
                mlflow.log_metric(metric_name, value)
            
            # Register model in MLflow Model Registry
            model_details = mlflow.register_model(
                model_uri=f"runs:/{mlflow.active_run().info.run_id}/model",
                name=MODEL_NAME
            )
            
            # Log model version
            mlflow.log_param("model_version", model_details.version)
            
            logger.info(f"Model training completed successfully. Best algorithm: {best_algo}")
            logger.info(f"Model registered with version: {model_details.version}")
            
            # Return the model URI for downstream tasks
            return f"runs:/{mlflow.active_run().info.run_id}/model"
    except Exception as e:
        logger.error(f"Error in train_model: {str(e)}")
        raise

def evaluate_model(**context):
    """Evaluate model performance and log metrics to MLflow."""
    try:
        with mlflow.start_run(run_name="model_evaluation", experiment_id=experiment_id, nested=True):
            # Get model URI from previous task
            model_uri = context['ti'].xcom_pull(task_ids='train_model')
            if not model_uri:
                raise ValueError("No model URI found from train_model task")
                
            # Load model from MLflow
            model = mlflow.sklearn.load_model(model_uri)
            
            weighted_path = context['ti'].xcom_pull(task_ids='preprocess_data')
            if not os.path.exists(weighted_path):
                raise FileNotFoundError(f"Preprocessed data file not found at {weighted_path}")
                
            X_weight = pd.read_csv(weighted_path)
            
            # --- Log training data size --- #
            training_data_size = len(X_weight)
            mlflow.log_metric("training_data_size", training_data_size)
            logger.info(f"Logged training_data_size: {training_data_size}")
            # --- End Log training data size --- #

            # Make predictions
            labels = model.predict(X_weight)
            
            # Calculate metrics
            silhouette = float(silhouette_score(X_weight, labels))
            calinski = float(calinski_harabasz_score(X_weight, labels))
            davies = float(davies_bouldin_score(X_weight, labels))
            
            # Log metrics to MLflow
            mlflow.log_metric("silhouette_score", silhouette)
            mlflow.log_metric("calinski_harabasz_score", calinski)
            mlflow.log_metric("davies_bouldin_score", davies)
            
            # Log cluster statistics
            cluster_sizes = pd.Series(labels).value_counts()
            for cluster, size in cluster_sizes.items():
                mlflow.log_metric(f"cluster_{int(cluster)}_size", int(size))
            
            # Log model parameters
            if hasattr(model, 'n_clusters'):
                mlflow.log_param("n_clusters", int(model.n_clusters))
            if hasattr(model, 'n_components'):
                mlflow.log_param("n_components", int(model.n_components))
            
            logger.info(f"Model evaluation - Silhouette: {silhouette:.4f}, Calinski: {calinski:.4f}, Davies: {davies:.4f}")
            return True
    except Exception as e:
        logger.error(f"Error in model evaluation: {str(e)}")
        raise

def monitor_model_performance(**context):
    """Monitor model performance and alert if degradation detected."""
    try:
        with mlflow.start_run(run_name="performance_monitoring", experiment_id=experiment_id, nested=True):
            # Get the latest model metrics
            client = MlflowClient()
            runs = client.search_runs(
                experiment_ids=[experiment_id],
                filter_string="tags.mlflow.runName = 'model_evaluation'",
                max_results=2
            )
            
            if not runs:
                logger.warning("No model evaluation runs found. Skipping performance monitoring.")
                mlflow.log_metric("monitoring_skipped", 1.0)
                return True
                
            latest_run = runs[0]
            
            # Check if required metrics exist in the latest run
            required_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
            if not all(metric in latest_run.data.metrics for metric in required_metrics):
                logger.warning("Latest run is missing required metrics. Skipping performance monitoring.")
                mlflow.log_metric("monitoring_skipped", 1.0)
                return True
            
            # Get training data size
            training_data_size = latest_run.data.metrics.get('training_data_size', 0)
            
            current_metrics = {
                'silhouette': float(latest_run.data.metrics['silhouette_score']),
                'calinski': float(latest_run.data.metrics['calinski_harabasz_score']),
                'davies': float(latest_run.data.metrics['davies_bouldin_score'])
            }
            
            # Log current metrics
            for metric, value in current_metrics.items():
                mlflow.log_metric(f"current_{metric}", value)
            
            # Calculate weighted score based on clustering metrics and data size
            weights = {
                'silhouette': 0.5,  # Higher weight for silhouette as it's most important for clustering
                'calinski': 0.3,
                'davies': 0.2
            }
            
            # Calculate base score (higher is better for silhouette and calinski, lower is better for davies)
            base_score = (
                weights['silhouette'] * current_metrics['silhouette'] +
                weights['calinski'] * current_metrics['calinski'] +
                weights['davies'] * (1 - current_metrics['davies'])  # Invert davies score as lower is better
            )
            
            # Calculate data size factor (normalize to 0-1 range)
            # Assuming 1000 samples is a good baseline for clustering
            data_size_factor = min(training_data_size / 1000, 1.0)
            
            # Calculate final weighted score
            final_score = base_score * (0.7 + 0.3 * data_size_factor)
            
            # Log the weighted score and data size
            mlflow.log_metric("weighted_score", final_score)
            mlflow.log_metric("training_data_size", training_data_size)
            mlflow.log_metric("data_size_factor", data_size_factor)
            
            # Only compare with previous run if it exists and has required metrics
            if len(runs) > 1:
                previous_run = runs[1]
                if all(metric in previous_run.data.metrics for metric in required_metrics):
                    previous_metrics = {
                        'silhouette': float(previous_run.data.metrics['silhouette_score']),
                        'calinski': float(previous_run.data.metrics['calinski_harabasz_score']),
                        'davies': float(previous_run.data.metrics['davies_bouldin_score'])
                    }
                    
                    # Check for significant degradation
                    threshold = 0.1  # 10% degradation threshold
                    for metric in current_metrics:
                        if metric == 'davies':  # For davies, higher is worse
                            if current_metrics[metric] > previous_metrics[metric] * (1 + threshold):
                                logger.warning(f"Significant degradation detected in {metric}")
                                mlflow.log_metric(f"{metric}_degradation", 1.0)
                            else:
                                mlflow.log_metric(f"{metric}_degradation", 0.0)
                        else:  # For silhouette and calinski, higher is better
                            if current_metrics[metric] < previous_metrics[metric] * (1 - threshold):
                                logger.warning(f"Significant degradation detected in {metric}")
                                mlflow.log_metric(f"{metric}_degradation", 1.0)
                            else:
                                mlflow.log_metric(f"{metric}_degradation", 0.0)
                else:
                    logger.warning("Previous run is missing required metrics. Skipping degradation comparison.")
                    mlflow.log_metric("previous_run_metrics_missing", 1.0)
            else:
                logger.info("No previous run found for comparison. This is the first evaluation.")
                mlflow.log_metric("first_evaluation", 1.0)
            
            logger.info(f"Model performance monitoring completed. Weighted score: {final_score:.4f}")
            logger.info(f"Training data size: {training_data_size}")
            return True
            
    except Exception as e:
        logger.error(f"Error in performance monitoring: {str(e)}")
        raise

def deploy_model(**context):
    """Deploy the best model."""
    try:
        # Get model URI from train_model task
        model_uri = context['ti'].xcom_pull(task_ids='train_model')
        if not model_uri:
            raise ValueError("No model URI found from train_model task")
            
        # Load model from MLflow
        model = mlflow.sklearn.load_model(model_uri)
        
        # Save model to production location
        prod_path = "/app/models"
        
        # Clean up existing model files if they exist
        if os.path.exists(prod_path):
            logger.info(f"Cleaning up existing model files in {prod_path}")
            for item in os.listdir(prod_path):
                item_path = os.path.join(prod_path, item)
                if os.path.isfile(item_path):
                    os.remove(item_path)
                elif os.path.isdir(item_path):
                    import shutil
                    shutil.rmtree(item_path)
        
        # Create fresh directory
        os.makedirs(prod_path, exist_ok=True)
        
        # Save model to production location
        mlflow.sklearn.save_model(model, prod_path)
                        
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

evaluate_task = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)

# Register model using MLflow plugin operator
register_model = MLflowModelOperator(
    task_id='register_model',
    mlflow_tracking_uri=MLFLOW_TRACKING_URI,
    model_name=MODEL_NAME,
    experiment_name=EXPERIMENT_NAME,
    dag=dag
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
create_experiment >> extract_task >> validate_task >> preprocess_task >> train_task >> evaluate_task >> register_model >> monitor_task >> deploy_task