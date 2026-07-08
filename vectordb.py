import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from google import genai
from google.genai import types
from tools import Answer
import os

chroma_client = chromadb.PersistentClient(path="./chroma")
collection = chroma_client.create_collection(name="summarizing_agent_rag")

emb_client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))


# function for flattening the briefs of the meeting
def build_documents(answer: Answer, meeting_id: str, date: str) -> list[dict]:
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