import os
import mlflow
from datetime import datetime

class MLflowTracker:
    def __init__(self):
        # Set MLflow tracking URI - gunakan SQLite local tracking by default
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
        
        # Buat experiment untuk tracking
        self.experiment_name = "document_analysis"
        if mlflow.get_experiment_by_name(self.experiment_name) is None:
            mlflow.create_experiment(self.experiment_name)
        mlflow.set_experiment(self.experiment_name)

    def log_document_metrics(self, doc_id: str, metrics: dict):
        """
        Log metrics untuk analisis dokumen
        """
        with mlflow.start_run(run_name=f"doc_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log metrics
            mlflow.log_metrics(metrics)
            
            # Log tags
            mlflow.set_tags({
                "document_id": doc_id,
                "processing_date": datetime.now().strftime("%Y-%m-%d")
            })

    def log_clustering_result(self, doc_id: str, cluster_result: dict):
        """
        Log hasil clustering
        """
        with mlflow.start_run(run_name=f"cluster_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_metrics({
                "cluster_value": cluster_result["cluster"],
                "confidence_score": cluster_result.get("confidence", 0.0)
            })
            
            # Log parameter clustering
            mlflow.log_params({
                "n_sentences": cluster_result["n_sentences"],
                "n_pages": cluster_result["n_pages"],
                "time_diff": cluster_result["time_diff"]
            })

    def log_plagiarism_check(self, doc_id: str, plagiarism_results: dict):
        """
        Log hasil pengecekan plagiarism
        """
        with mlflow.start_run(run_name=f"plagiarism_{doc_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log metrics
            max_similarity = max(plagiarism_results.values()) if plagiarism_results else 0.0
            mlflow.log_metrics({
                "max_similarity_score": max_similarity,
                "num_similar_docs": len(plagiarism_results)
            })
            
            # Log matched documents as parameters
            mlflow.log_params({
                f"match_{i}": f"{doc}:{score}"
                for i, (doc, score) in enumerate(plagiarism_results.items())
            })

    def log_model(self, model, model_name: str, params: dict = None):
        """
        Log model machine learning
        """
        with mlflow.start_run(run_name=f"model_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            if params:
                mlflow.log_params(params)
            mlflow.sklearn.log_model(model, model_name)
