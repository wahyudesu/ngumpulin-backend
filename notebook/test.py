import os
from dotenv import load_dotenv

load_dotenv()

print("SUPABASE_URL:", os.getenv("SUPABASE_URL"))
print("SUPABASE_KEY:", os.getenv("SUPABASE_KEY"))
print("PINECONE_API_KEY", os.getenv("PINECONE_API_KEY"))