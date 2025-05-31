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

class MLflowModelOperator(MLflowOperator):
    """
    Operator for MLflow model operations.
    """
    
    @apply_defaults
    def __init__(
        self,
        model_path: str,
        model_name: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.model_path = model_path
        self.model_name = model_name

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the operator."""
        super().execute(context)
        try:
            with mlflow.start_run():
                model = mlflow.sklearn.load_model(self.model_path)
                mlflow.sklearn.log_model(
                    model,
                    self.model_name,
                    registered_model_name=self.model_name
                )
                logger.info(f"Successfully registered model {self.model_name}")
        except Exception as e:
            logger.error(f"Error registering model: {str(e)}")
            raise

class MLflowExperimentOperator(MLflowOperator):
    """
    Operator for MLflow experiment operations.
    """
    
    @apply_defaults
    def __init__(
        self,
        experiment_name: str,
        *args: Any,
        **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name

    def execute(self, context: Dict[str, Any]) -> None:
        """Execute the operator."""
        super().execute(context)
        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            if experiment is None:
                experiment_id = mlflow.create_experiment(self.experiment_name)
                logger.info(f"Created experiment {self.experiment_name} with ID {experiment_id}")
            else:
                logger.info(f"Experiment {self.experiment_name} already exists")
        except Exception as e:
            logger.error(f"Error managing experiment: {str(e)}")
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