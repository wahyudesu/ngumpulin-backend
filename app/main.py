import os
import re
import json
import numpy as np
import tempfile
import shutil
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeEmbeddings
from gliner import GLiNER
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
import logging
import mlflow
import mlflow.sklearn
from app.routers.agent import generate_feedback, AssignmentRequest
from app.utils.metrics import setup_metrics
from app.core.exceptions import ModelLoadError
from contextlib import asynccontextmanager
from pydantic import BaseModel
from typing import List, Dict, Any
from mlflow.tracking import MlflowClient

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Constants
MODEL_NAME = "document-clustering"
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# Initialize MLflow
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("document-processing")

def load_best_model():
    """Load the best model based on weighted score considering clustering metrics and training data size."""
    temp_dir = None # Initialize temp_dir
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("document-processing")
        
        if experiment is None:
            logger.warning("Experiment 'document-processing' not found, trying to load local model")
            # Instead of returning local model directly here, raise an error to trigger fallback in lifespan
            raise ModelLoadError("MLflow experiment not found")
            
        # Get all runs and calculate weighted scores
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.mlflow.runName = 'model_evaluation'"
        )
        
        if not runs:
            logger.warning("No model runs found, trying to load local model")
            # Instead of returning local model directly here, raise an error to trigger fallback in lifespan
            raise ModelLoadError("No valid model runs found in MLflow")
            
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
            # Ensure metrics are floats
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
            # Assuming 1000 is the baseline
            data_size_factor = min(training_data_size / 1000.0, 1.0)
            
            # Calculate final weighted score
            final_score = base_score * (0.7 + 0.3 * data_size_factor)
            
            if final_score > best_score:
                best_score = final_score
                best_run = run
        
        if best_run is None:
            logger.warning("No valid model runs found with required metrics, trying to load local model")
            # Instead of returning local model directly here, raise an error to trigger fallback in lifespan
            raise ModelLoadError("No valid model runs found with required metrics in MLflow")
            
        # --- Download the model artifact ---  
        temp_dir = tempfile.mkdtemp() # Create a temporary directory
        artifact_path = "model" # The artifact path logged in train_model
        local_path = client.download_artifacts(best_run.info.run_id, artifact_path, dst_path=temp_dir)
        
        # Load the model from the downloaded path
        model = mlflow.sklearn.load_model(local_path)
        
        # Log model details
        logger.info(f"Downloaded and loaded model from run {best_run.info.run_id} to {local_path}")
        logger.info(f"Training data size: {best_run.data.metrics.get('training_data_size', 0)}")
        
        return model
        
    except Exception as e:
        logger.warning(f"Error loading model from MLflow: {str(e)}") # Keep warning, but re-raise to trigger fallback
        raise ModelLoadError(f"Failed to load model from MLflow: {str(e)}") # Re-raise the exception
        
    finally:
        # Clean up the temporary directory
        if temp_dir and os.path.exists(temp_dir):
            try:
                shutil.rmtree(temp_dir)
                logger.info(f"Cleaned up temporary directory {temp_dir}")
            except Exception as cleanup_e:
                logger.warning(f"Error cleaning up temporary directory {temp_dir}: {str(cleanup_e)}")

def load_local_model():
    """Load model from local directory if MLflow is not available."""
    try:
        local_model_path = "/app/models/local_model.pkl" # Assuming the model is saved as local_model.pkl by Airflow
        # Check if the model directory exists and contains a file, or if the pkl file exists
        if os.path.exists(local_model_path) and os.path.isfile(local_model_path):
            logger.info(f"Loading model from local file {local_model_path}")
            with open(local_model_path, 'rb') as f:
                model = pickle.load(f)
            return model
        elif os.path.exists("/app/models") and os.path.isdir("/app/models") and any(os.path.isfile(os.path.join("/app/models", f)) for f in os.listdir("/app/models")):
            # If /app/models exists and contains files, it might be the MLflow saved model structure
            mlflow_local_path = "/app/models" # Assuming the entire MLflow model structure is copied here
            logger.info(f"Loading model from local MLflow structure at {mlflow_local_path}")
            model = mlflow.sklearn.load_model(mlflow_local_path)
            return model
        else:
            logger.warning("No local model found at expected locations (/app/models/local_model.pkl or /app/models)")
            raise ModelLoadError("No model available locally")
    except Exception as e:
        logger.error(f"Error loading local model: {str(e)}")
        raise ModelLoadError(f"Failed to load local model: {str(e)}")

class PredictionInput(BaseModel):
    sentences: int
    page: int
    timing: int
    plagiarism: float

class PredictionOutput(BaseModel):
    cluster: int
    confidence: float
    metadata: Dict[str, Any]

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application...")
    try:
        # First, try loading the local model
        model = load_local_model()
        logger.info("Local model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load local model: {str(e)}")
        # If local model fails, try loading from MLflow as a fallback
        try:
            model = load_best_model()
            logger.info("MLflow model loaded successfully")
        except Exception as mlflow_e:
            logger.error(f"Failed to load model from MLflow: {str(mlflow_e)}")
            model = None # Ensure model is None if both fail
    yield
    # Shutdown
    logger.info("Shutting down the application...")

app = FastAPI(
    title="FastAPI ML Document Processing",
    description="Document processing API with ML capabilities",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)
setup_metrics(app)

# Initialize services
supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
labels = ["Name", "ID"]
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# Main route
@app.get("/")
def read_root():
    return {"message": "FastAPI ML Document Processing API", "version": "1.0.0"}

# Health check endpoint
@app.get("/health")
def health_check():
    try:
        # Check MLflow connection
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        client = mlflow.tracking.MlflowClient()
        try:
            # Try to get experiment instead of list_experiments
            experiment = client.get_experiment_by_name("document-processing")
            mlflow_status = "connected" if experiment is not None else "connecting"
        except Exception as mlflow_error:
            logger.warning(f"MLflow connection issue: {str(mlflow_error)}")
            mlflow_status = "connecting"
        
        # Try to load model, but don't fail if not available yet
        try:
            model = load_best_model()
            model_status = "loaded"
        except Exception as model_error:
            logger.warning(f"Model not loaded yet: {str(model_error)}")
            model_status = "loading"
        
        # Check Supabase connection
        try:
            supabase.table("documents").select("count").limit(1).execute()
            supabase_status = "connected"
        except Exception as supabase_error:
            logger.warning(f"Supabase connection issue: {str(supabase_error)}")
            supabase_status = "connecting"
        
        return {
            "status": "healthy",
            "components": {
                "mlflow": mlflow_status,
                "model": model_status,
                "supabase": supabase_status
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "components": {
                "mlflow": "error" if "mlflow" in str(e).lower() else "unknown",
                "model": "error" if "model" in str(e).lower() else "unknown",
                "supabase": "error" if "supabase" in str(e).lower() else "unknown"
            }
        }, 503

# Upload route
@app.post("/upload")
async def upload_file(uuid: str = Form(...), file_url: str = Form(...)):
    with mlflow.start_run(run_name=f"document_processing_{uuid}"):
        try:
            # Log parameters
            mlflow.log_param("uuid", uuid)
            mlflow.log_param("file_url", file_url)

            # Parse PDF
            loader = PyPDFLoader(file_url)
            documents = loader.load()
            page_count = len(documents)
            full_text = " ".join(doc.page_content for doc in documents)
            sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if s.strip()])
            
            # Log basic metrics
            mlflow.log_metric("page_count", page_count)
            mlflow.log_metric("sentence_count", sentence_count)

            # Chunking and embedding
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            chunks = text_splitter.split_documents(documents)
            markdown_content = "\n\n".join(doc.page_content for doc in chunks)
            vector = embeddings.embed_documents([markdown_content])[0]

            # Entity extraction
            first_chunk = chunks[0].page_content if chunks else ""
            entities = model.predict_entities(first_chunk, labels)
            extracted_data = {"Name": "", "ID": ""}
            for entity in entities:
                extracted_data[entity["label"]] = entity["text"]

            # Get current record
            current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
            if not current_record.data:
                raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")
            
            folder = current_record.data[0]["folder"]
            uploaded_date = current_record.data[0]["uploadedDate"]
            deadline = current_record.data[0]["deadline"]
            previous_records = supabase.table("documents").select("*").eq("folder", folder).lt("uploadedDate", uploaded_date).execute().data
            plagiarism_results = {}

            # Plagiarism detection
            if previous_records:
                previous_embeddings = []
                for record in previous_records:
                    try:
                        embedding = (
                            json.loads(record["embedding"]) if isinstance(record["embedding"], str)
                            else record["embedding"]
                        )
                        previous_embeddings.append([float(x) for x in embedding])
                    except Exception:
                        continue

                if previous_embeddings:
                    current_embedding = vector if isinstance(vector, list) else [float(x) for x in json.loads(vector)]
                    similarities = cosine_similarity(np.array([current_embedding]), np.array(previous_embeddings))[0]
                    similarity_list = [
                        (r["nameStudent"] or "Unknown", float(sim) if isinstance(sim, (int, float)) else 0.0)
                        for r, sim in zip([r for r in previous_records if r.get("embedding")], similarities)
                    ]
                    top_2 = sorted(similarity_list, key=lambda x: x[1], reverse=True)[:2]
                    plagiarism_results = dict(top_2)

            plagiarism_score = max(plagiarism_results.values()) if plagiarism_results else 0.0
            mlflow.log_metric("plagiarism_score", plagiarism_score)

            # Time difference
            deadline_dt = datetime.fromisoformat(deadline)
            uploaded_date_dt = datetime.fromisoformat(uploaded_date)
            time_diff = (deadline_dt - uploaded_date_dt).total_seconds() / 3600

            # Clustering
            df = pd.DataFrame({
                'sentences': [sentence_count],
                'page': [page_count],
                'timing': [time_diff],
                'plagiarism': [plagiarism_score]
            })
            scaled_data = StandardScaler().fit_transform(df)
            cluster = model.fit_predict(scaled_data)[0]

            # Update DB
            response = supabase.table("documents").update({
                "nameStudent": extracted_data["Name"] or "null",
                "NRP": extracted_data["ID"],
                "isiTugas": markdown_content,
                "embedding": vector.tolist() if isinstance(vector, np.ndarray) else vector,
                "page": page_count,
                "sentences": sentence_count,
                "plagiarism": plagiarism_results,
                "clustering": float(cluster)
            }).eq("id", uuid).execute()

            if not response.data:
                raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")

            return {
                "message": "File processed and record updated successfully.",
                "extracted_entities": extracted_data,
                "vector": vector,
                "page_count": page_count,
                "sentence_count": sentence_count,
                "plagiarism_results": plagiarism_results
            }

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# Route baru untuk memproses feedback assignment
@app.post("/agent-feedback")
async def agent_feedback_endpoint(uuid: str = Form(...)):
    with mlflow.start_run(run_name=f"agent_feedback_{uuid}"):
        try:
            mlflow.log_param("uuid", uuid)
            
            # Get document data
            current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
            if not current_record.data:
                raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")
            
            assignment_content = current_record.data[0]["isiTugas"]
            folder = current_record.data[0]["folder"]

            # Get folder information
            folder_record = supabase.table("folders").select("nameAssignment", "description").eq("nameAssignment", folder).execute()
            if not folder_record.data:
                raise HTTPException(status_code=404, detail=f"No folder found with nameAssignment: {folder}")

            title = folder_record.data[0]["nameAssignment"]
            description = folder_record.data[0]["description"]

            # Log parameters
            mlflow.log_param("title", title)
            mlflow.log_param("content_length", len(assignment_content))

            # Generate feedback
            persona = "Provide feedback in a formal and constructive tone suitable for academic purposes."
            payload = AssignmentRequest(
                assignment_meta={
                    "title": title,
                    "description": description
                },
                assignment_content=assignment_content
            )

            feedback_result = generate_feedback(payload)
            
            # Log metrics
            mlflow.log_metric("feedback_generated", 1)
            
            return feedback_result

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    try:
        model = load_best_model()
        
        # Create input array
        X = np.array([[input_data.sentences, input_data.page, input_data.timing, input_data.plagiarism]])
        
        # Manual min-max scaling
        # Assuming these are the min and max values from your training data
        min_values = np.array([0, 0, 0, 0])  # Minimum values for each feature
        max_values = np.array([100, 100, 100, 100])  # Maximum values for each feature
        
        # Apply min-max scaling
        X_scaled = (X - min_values) / (max_values - min_values)
        
        # Apply feature weights (same as in DAG)
        weights = np.array([0.5, 0.5, 1.5, 4.5])
        X_weighted = X_scaled * weights
        
        # Make prediction
        cluster = int(model.predict(X_weighted)[0])
        
        # Calculate confidence (distance to cluster center)
        if hasattr(model, 'cluster_centers_'):
            centers = model.cluster_centers_
            distances = np.linalg.norm(X_weighted - centers[cluster])
            confidence = float(1 / (1 + distances))  # Convert distance to confidence score
        else:
            confidence = 1.0
            
        # Get model metadata
        metadata = {
            "model_type": type(model).__name__,
            "n_clusters": getattr(model, 'n_clusters_', getattr(model, 'n_components_', None)),
            "feature_weights": weights.tolist(),
            "scaling": {
                "type": "min-max",
                "min_values": min_values.tolist(),
                "max_values": max_values.tolist()
            }
        }
        
        return PredictionOutput(
            cluster=cluster,
            confidence=confidence,
            metadata=metadata
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/model-info")
def get_model_info():
    try:
        model = load_best_model()
        metadata = {
            "model_type": type(model).__name__,
            "n_clusters": getattr(model, 'n_clusters_', getattr(model, 'n_components_', None)),
            "feature_weights": [0.5, 0.5, 1.5, 4.5],
            "scaling": {
                "type": "min-max",
                "min_values": [0, 0, 0, 0],
                "max_values": [100, 100, 100, 100]
            }
        }
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting model info: {str(e)}")