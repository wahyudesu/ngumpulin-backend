import os
import re
import json
import numpy as np
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
# from routers.agent import generate_feedback, AssignmentRequest
from prometheus_fastapi_instrumentator import Instrumentator
import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd
import warnings
import logging
import mlflow
import mlflow.sklearn
from app.utils.metrics import setup_metrics
from contextlib import asynccontextmanager
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize MLflow
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
mlflow.set_experiment("document-processing")

# Load the best model from MLflow
def load_best_model():
    try:
        # Get the best run based on metrics
        runs = mlflow.search_runs(
            experiment_ids=[mlflow.get_experiment_by_name("document-processing").experiment_id],
            filter_string="metrics.silhouette_score > 0",
            order_by=["metrics.silhouette_score DESC"]
        )
        
        if runs.empty:
            logger.warning("No model found in MLflow, loading from local file")
            model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
            with open(model_path, 'rb') as file:
                return pickle.load(file)
        
        # Get the latest production model from MLflow Model Registry
        client = mlflow.tracking.MlflowClient()
        try:
            # Try to get the latest production model
            model_uri = f"models:/document-clustering/Production"
            logger.info(f"Loading production model from MLflow: {model_uri}")
            return mlflow.sklearn.load_model(model_uri)
        except Exception as e:
            logger.warning(f"Could not load production model: {str(e)}")
            # Fallback to best run
            best_run = runs.iloc[0]
            model_uri = f"runs:/{best_run.run_id}/model"
            logger.info(f"Loading best run model from MLflow: {model_uri}")
            return mlflow.sklearn.load_model(model_uri)
    except Exception as e:
        logger.error(f"Error loading model from MLflow: {str(e)}")
        logger.info("Falling back to local model")
        model_path = os.path.join(os.path.dirname(__file__), 'best_model.pkl')
        with open(model_path, 'rb') as file:
            return pickle.load(file)

# Load the model
loaded_model = load_best_model()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting up the application...")
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
    return {"status": "healthy"}

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
            cluster = loaded_model.fit_predict(scaled_data)[0]

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

# # Model request untuk agent feedback
# class AssignmentMetaModel(BaseModel):
#     title: str
#     description: str

# class AssignmentRequest(BaseModel):
#     assignment_meta: AssignmentMetaModel
#     assignment_content: str

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
            if not folder_record:
                raise HTTPException(status_code=404, detail=f"No folder found with nameAssignment: {folder}")

            title = folder_record.data[0]["nameAssignment"]
            description = folder_record.data[0]["description"]

            # Log parameters
            mlflow.log_param("title", title)
            mlflow.log_param("content_length", len(assignment_content))

            # Generate feedback
            persona = "Provide feedback in a formal and constructive tone suitable for academic purposes."
            payload = AssignmentRequest(
                title=title,
                description=description,
                content=assignment_content,
                persona=persona
            )

            feedback_result = generate_feedback(payload)
            
            # Log metrics
            mlflow.log_metric("feedback_generated", 1)
            
            return feedback_result

        except Exception as e:
            mlflow.log_param("error", str(e))
            raise HTTPException(status_code=500, detail=str(e))

# @app.post("/agent-feedback")
# async def agent_feedback_endpoint(uuid: str = Form(...)):
#     try:
#         # Get document data by UUID
#         current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
#         if not current_record.data:
#             raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")
        
#         # Extract assignment content from the current record
#         assignment_content = current_record.data[0]["isiTugas"]
#         folder = current_record.data[0]["folder"]

#         # Get previous records in the same folder
#         tugas = supabase.table("documents").select("folder").eq("folder", folder).execute().data

#         deadline = tugas.data["deadline"]
        
#         # Process feedback
#         feedback_result = generate_feedback(
#             assignment_meta={"title": "Assignment", "description": "Student Submission"},
#             assignment_content=assignment_content
#         )
#         return feedback_result
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))