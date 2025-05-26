import os
import re
import json
import numpy as np
from fastapi import FastAPI, HTTPException, Form
from supabase import create_client
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeEmbeddings
from gliner import GLiNER
from dotenv import load_dotenv
# from pinecone import Pinecone
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from routers.agent import generate_feedback, AssignmentRequest

import pickle
from sklearn.preprocessing import StandardScaler
import pandas as pd

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

app = FastAPI()

supabase = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
model = GLiNER.from_pretrained("urchade/gliner_medium-v2.1")
labels = ["Name", "ID"]
# pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
embeddings = PineconeEmbeddings(model="multilingual-e5-large")

# Load the saved model
with open('best_model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Main route
@app.get("/")
def read_root():
    return {"message": "Hello World"}

# Upload route
@app.post("/upload")
async def upload_file(uuid: str = Form(...), file_url: str = Form(...)):
    try:
        # Parse pdf
        loader = PyPDFLoader(file_url)
        documents = loader.load()
        page_count = len(documents)
        full_text = " ".join(doc.page_content for doc in documents)
        sentence_count = len([s for s in re.split(r'[.!?]+', full_text) if s.strip()])

        # chunking and embedding to vector DB
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

        # Get data specific documents
        current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
        if not current_record.data:
            raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")

        # Plagiarism detection
        folder = current_record.data[0]["folder"]
        uploaded_date = current_record.data[0]["uploadedDate"]
        previous_records = supabase.table("documents").select("*").eq("folder", folder).lt("uploadedDate", uploaded_date).execute().data
        plagiarism_results = {}

        if previous_records:
            previous_embeddings = []
            for record in previous_records:
                if record.get("embedding"):
                    try:
                        if isinstance(record["embedding"], str):
                            embedding = [float(x) for x in json.loads(record["embedding"])]
                        else:
                            embedding = [float(x) for x in record["embedding"]]
                        previous_embeddings.append(embedding)
                    except (json.JSONDecodeError, ValueError, TypeError):
                        print(f"Invalid embedding in record: {record}")
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
                
        # Calculate time difference in minutes between deadline and uploaded date
        deadline = current_record.data[0]["deadline"]
        uploaded_date = current_record.data[0]["uploadedDate"]
        
        # deadline_dt = datetime.strptime(deadline, "%d-%m-%Y %H:%M")
        # uploaded_date_dt = datetime.strptime(deadline, "%d-%m-%Y %H:%M")
        
        # Convert to datetime objects
        deadline_dt = datetime.fromisoformat(deadline)
        uploaded_date_dt = datetime.fromisoformat(uploaded_date)
        
        time_diff = (deadline_dt - uploaded_date_dt).total_seconds() / 3600
        
        plagiarism_score = (
            max(plagiarism_results.values()) if plagiarism_results else 0.0
        )

        # Clustering
        data_prediction = pd.DataFrame({
            'sentences': [sentence_count],
            'page': [page_count],
            'timing': [time_diff],
            'plagiarism': [plagiarism_score]
        })

        data_prediction = StandardScaler().fit_transform(data_prediction)
        clustering = loaded_model.fit_predict(data_prediction)
        clustering_value = clustering[0] if isinstance(clustering, (list, np.ndarray)) else clustering

        # Update to database
        response = supabase.table("documents").update({
            "nameStudent": extracted_data["Name"] or "null",
            "NRP": extracted_data["ID"],
            "isiTugas": markdown_content,
            "embedding": vector.tolist() if isinstance(vector, np.ndarray) else vector,
            "page": page_count,
            "sentences": sentence_count,
            "plagiarism": plagiarism_results,
            "clustering": float(clustering_value)
        }).eq("id", uuid).execute()

        # Error handling
        if not response.data:
            raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")
        
        # Variable return
        return {
            "message": "File processed and record updated successfully in Supabase.",
            "extracted_entities": extracted_data,
            "vector": vector,
            "page_count": page_count,
            "sentence_count": sentence_count,
            "plagiarism_results": plagiarism_results
        }
    
    except Exception as e:
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
    try:
        # Ambil record dari tabel documents berdasarkan UUID
        current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
        if not current_record.data:
            raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")
        
        # Ekstrak isi tugas dan nama folder
        assignment_content = current_record.data[0]["isiTugas"]
        folder = current_record.data[0]["folder"]

        # Ambil title dan deskripsi dari tabel folders
        folder_record = supabase.table("folders").select("nameAssignment", "description").eq("nameAssignment", folder).execute()
        if not folder_record:
            raise HTTPException(status_code=404, detail=f"No folder found with nameAssignment: {folder}")

        title = folder_record.data[0]["nameAssignment"]
        description = folder_record.data[0]["description"]

        logging.info(f"OUTPUTKONTOL: {title}")
        logging.info(f"OUTPUT: {description}")

        # Persona untuk feedback
        persona = "Provide feedback in a formal and constructive tone suitable for academic purposes."

        # Buat request payload
        payload = AssignmentRequest(
            title=title,
            description=description,
            content=assignment_content,
            persona=persona
        )

        # Jalankan agent
        feedback_result = generate_feedback(payload)
        return feedback_result

    except Exception as e:
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

