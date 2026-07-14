from langchain.tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from langchain.chat_models import init_chat_model
from vectordb import emb_client, collection
import os
import whisper

load_dotenv()

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
api_key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model(
    "google_genai:gemini-3.1-flash-lite",
)

@tool
def transcribe(file_path:str)->str:
    """Given the audio file path fetch it and transcribe it and return the transcription.
    Args:
        file_path (str): The audio file path that needs to be transcribed
    Returns: 
        transcription (str): The transcription of the audio file"""
    model = whisper.load_model('turbo')
    result = model.transcribe(file_path)
    transcription = result["text"]
    return transcription


class ActionItem(BaseModel):
    owner:str
    task:str

class OpenQuestion(BaseModel):
    question:str
    asked_by:str

class Risk(BaseModel):
    risk:str

class Summarizing_Answer(BaseModel):
    title:str
    summary:str
    action_items:list[ActionItem]
    open_questions:list[OpenQuestion]
    risks: Optional[Risk]


summarizing_model_with_structured_output = model.with_structured_output(Summarizing_Answer)



# tools for rag agent 

# function for flattening the briefs of the meeting
def build_documents(answer: Summarizing_Answer, meeting_id: str, date: str) -> list[dict]:
    docs = []
    
    # Summary level doc
    docs.append({
        "text": f"Meeting: {answer.title}\nSummary: {answer.summary}",
        "metadata": {"meeting_id": meeting_id, "date": date, "type": "summary"}
    })
    
    # One doc per action item
    for item in answer.action_items:
        docs.append({
            "text": f"Action item from meeting '{answer.title}' ({date}): {item.description}",
            "metadata": {"meeting_id": meeting_id, "date": date, "type": "action_item"}
        })
    
    # One doc per open question
    for q in answer.open_questions:
        docs.append({
            "text": f"Open question from meeting '{answer.title}' ({date}): {q.question}",
            "metadata": {"meeting_id": meeting_id, "date": date, "type": "open_question"}
        })
    
    # Risk doc
    if answer.risks:
        docs.append({
            "text": f"Risk noted in meeting '{answer.title}' ({date}): {answer.risks}",
            "metadata": {"meeting_id": meeting_id, "date": date, "type": "risk"}
        })
    return docs

#function for embedding summary, action items, risks and open questions of a meeting 
def embed_doc(docs:list[dict]):
    texts = [d["text"] for d in docs]
    result = emb_client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=texts,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
    )
    for doc, embedding in zip(docs, result.embeddings):
        doc["embedding"] = embedding.values
    
    return docs

#function to store the document in chroma db
def store(docs:list[dict]):
    docs = embed_doc(docs)
    collection.add(
        ids=[d['id'] for d in docs],
        documents=[d['text'] for d in docs],
        embeddings=[d['embeddings'] for d in docs],
        metadatas=[d['metadata'] for d in docs]
    )

#function to embed user query
def embed_query(query:str):
    result = emb_client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY")
    )
    return result


#function to search for data related to user's query
def search(query:str):
    result = embed_query(query)
    result = collection.query(query_embeddings=result)
    data = result["documents"]
    return data

class Rag_Answer(BaseModel):
    response:str

rag_model_with_structured_output = model.with_structured_output(Rag_Answer)