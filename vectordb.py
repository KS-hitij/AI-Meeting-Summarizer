import chromadb
from google import genai
import os
from dotenv import load_dotenv

load_dotenv()

chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.get_or_create_collection(name="summarizing_agent_rag")

emb_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
