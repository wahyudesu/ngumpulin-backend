"""
Airflow plugins package for MLflow operators.
"""

from airflow.plugins_manager import AirflowPlugin
from mlflow_plugin import MLflowModelOperator, MLflowExperimentOperator, MLflowPlugin

# Register the plugin
__all__ = ['MLflowModelOperator', 'MLflowExperimentOperator', 'MLflowPlugin'] 