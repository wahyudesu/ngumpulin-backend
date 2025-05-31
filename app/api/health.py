from fastapi import APIRouter, HTTPException
from sqlalchemy import create_engine
import requests
import mlflow
from typing import Dict

router = APIRouter()

def check_postgres() -> Dict:
    try:
        engine = create_engine("postgresql://airflow:airflow@postgres/airflow")
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return {"status": "healthy", "message": "PostgreSQL is running"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"PostgreSQL error: {str(e)}"}

def check_mlflow() -> Dict:
    try:
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.get_experiment_by_name("default")
        return {"status": "healthy", "message": "MLflow is running"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"MLflow error: {str(e)}"}

def check_airflow() -> Dict:
    try:
        response = requests.get("http://airflow-webserver:8080/health")
        if response.status_code == 200:
            return {"status": "healthy", "message": "Airflow is running"}
        return {"status": "unhealthy", "message": f"Airflow returned status code {response.status_code}"}
    except Exception as e:
        return {"status": "unhealthy", "message": f"Airflow error: {str(e)}"}

@router.get("/health")
async def health_check():
    """
    Health check endpoint that verifies the status of all components
    """
    health_status = {
        "status": "healthy",
        "components": {
            "postgres": check_postgres(),
            "mlflow": check_mlflow(),
            "airflow": check_airflow()
        }
    }
    
    # Check if any component is unhealthy
    for component, status in health_status["components"].items():
        if status["status"] == "unhealthy":
            health_status["status"] = "unhealthy"
            break
    
    if health_status["status"] == "unhealthy":
        raise HTTPException(
            status_code=503,
            detail=health_status
        )
    
    return health_status 