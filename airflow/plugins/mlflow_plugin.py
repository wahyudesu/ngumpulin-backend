"""
MLflow plugin for Airflow.
This plugin provides operators for MLflow operations in Airflow DAGs.
"""

from airflow.plugins_manager import AirflowPlugin
from airflow.models import BaseOperator
from airflow.utils.decorators import apply_defaults
import mlflow
import mlflow.sklearn
from typing import Any, Dict, Optional
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class MLflowOperator(BaseOperator):
    """
    Base operator for MLflow operations.
    """
    
    @apply_defaults
    def __init__(
        self,
        mlflow_tracking_uri: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.mlflow_tracking_uri = mlflow_tracking_uri

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the operator."""
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)

class MLflowModelOperator(BaseOperator):
    """
    Operator to register and manage MLflow models
    """
    @apply_defaults
    def __init__(
        self,
        mlflow_tracking_uri,
        model_name,
        experiment_name="document-processing",
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.model_name = model_name
        self.experiment_name = experiment_name

    def execute(self, context):
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Get the client
            client = mlflow.tracking.MlflowClient()
            
            # Check if model exists
            try:
                client.get_registered_model(self.model_name)
            except Exception:
                # Create new model if it doesn't exist
                client.create_registered_model(self.model_name)
            
            # Get experiment
            experiment = client.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{self.experiment_name}' not found")
            
            # Get the latest run
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=f"tags.mlflow.runName = 'model_evaluation'",
                max_results=1
            )
            
            if not runs:
                raise ValueError("No model evaluation runs found")
            
            latest_run = runs[0]
            
            # Register new model version
            result = mlflow.register_model(
                model_uri=f"runs:/{latest_run.info.run_id}/model",
                name=self.model_name
            )
            
            # Transition to Production
            client.transition_model_version_stage(
                name=self.model_name,
                version=result.version,
                stage="Production"
            )
            
            # Archive previous versions
            for version in client.search_model_versions(f"name='{self.model_name}'"):
                if version.version != result.version and version.current_stage == "Production":
                    client.transition_model_version_stage(
                        name=self.model_name,
                        version=version.version,
                        stage="Archived"
                    )
            
            logger.info(f"Model registered successfully: {self.model_name} version {result.version}")
            return result.version
            
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise

class MLflowExperimentOperator(BaseOperator):
    """
    Operator to create MLflow experiments
    """
    @apply_defaults
    def __init__(
        self,
        mlflow_tracking_uri,
        experiment_name,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.mlflow_tracking_uri = mlflow_tracking_uri
        self.experiment_name = experiment_name

    def execute(self, context):
        try:
            # Set MLflow tracking URI
            mlflow.set_tracking_uri(self.mlflow_tracking_uri)
            
            # Create experiment if it doesn't exist
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
            else:
                experiment_id = experiment.experiment_id
            
            mlflow.set_experiment(experiment_id=experiment_id)
            logger.info(f"Experiment {self.experiment_name} (ID: {experiment_id}) is ready")
            return experiment_id
            
        except Exception as e:
            logger.error(f"Error creating experiment: {str(e)}")
            raise

class MLflowPlugin(AirflowPlugin):
    """MLflow plugin for Airflow."""
    
    name = "mlflow_plugin"
    operators = [MLflowModelOperator, MLflowExperimentOperator]
    hooks = []
    executors = []
    macros = []
    admin_views = []
    flask_blueprints = []
    menu_links = [] 