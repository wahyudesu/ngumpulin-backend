# from fastapi import APIRouter, HTTPException, Form
# from app.core.config import supabase
# from app.services.pdf_loader import load_pdf
# from app.services.entity_extraction import extract_entities
# from app.services.embedding import generate_embedding
# from app.services.plagiarism import check_plagiarism
# from app.services.clustering import predict_cluster

# router = APIRouter()

# @router.post("/")
# async def upload_file(uuid: str = Form(...), file_url: str = Form(...)):
#     try:
#         documents, page_count, sentence_count, markdown_content = load_pdf(file_url)
#         embedding_vector = generate_embedding(markdown_content)
#         extracted_data = extract_entities(documents)

#         current_record = supabase.table("documents").select("*").eq("id", uuid).execute()
#         if not current_record.data:
#             raise HTTPException(status_code=404, detail=f"No record found with uuid: {uuid}")

#         folder = current_record.data[0]["folder"]
#         uploaded_date = current_record.data[0]["uploadedDate"]
#         deadline = current_record.data[0]["deadline"]

#         previous_records = supabase.table("documents").select("*").eq("folder", folder).lt("uploadedDate", uploaded_date).execute().data
#         plagiarism_results = check_plagiarism(previous_records, embedding_vector)

#         from datetime import datetime
#         time_diff = (datetime.fromisoformat(deadline) - datetime.fromisoformat(uploaded_date)).total_seconds() / 3600
#         cluster_value = predict_cluster(sentence_count, page_count, time_diff, plagiarism_results)

#         response = supabase.table("documents").update({
#             "nameStudent": extracted_data["Name"] or "null",
#             "NRP": extracted_data["ID"],
#             "isiTugas": markdown_content,
#             "embedding": embedding_vector,
#             "page": page_count,
#             "sentences": sentence_count,
#             "plagiarism": plagiarism_results,
#             "clustering": cluster_value
#         }).eq("id", uuid).execute()

#         return {
#             "message": "File processed successfully.",
#             "extracted_entities": extracted_data,
#             "page_count": page_count,
#             "sentence_count": sentence_count,
#             "plagiarism_results": plagiarism_results
#         }

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
