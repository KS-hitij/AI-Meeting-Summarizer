from langchain.tools import tool
from vectordb import emb_client, collection
from google.genai import types
from models import SummarizingAnswer
import os
import whisper

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)


@tool
def transcribe(file_path: str) -> str:
    """Given the audio file path fetch it and transcribe it and return the transcription.
    Args:
        file_path (str): The audio file path that needs to be transcribed
    Returns:
        transcription (str): The transcription of the audio file"""
    model = whisper.load_model("turbo")
    result = model.transcribe(file_path)
    transcription = result["text"]
    return transcription


# tools for rag agent


@tool
def build_documents(
    answer: SummarizingAnswer, project_id: str, date: str
) -> list[dict]:
    """Given the summary of a meeting with it's project id and date build it in a list of data for embedding
    Args:
        answer (SummarizingAnswer): The summary of the meeting with it's action items, open ended questions and risks noted
        project_id (str): The id of the project to which the meeting belongs to
        date (str): The date on which the meeting took place
    Returns:
    docs (list[dict]): A list of dict ready for embedding"""

    docs = []

    # Summary level doc
    docs.append(
        {
            "text": f"Meeting: {answer.title}\nSummary: {answer.summary}",
            "metadata": {"project_id": project_id, "date": date, "type": "summary"},
            "id": f"{project_id}-summary",
        }
    )

    # One doc per action item
    for i, item in enumerate(answer.action_items):
        docs.append(
            {
                "text": f"Action item from meeting '{answer.title}' ({date}): {item.task} (Owner: {item.owner})",
                "metadata": {
                    "project_id": project_id,
                    "date": date,
                    "type": "action_item",
                },
                "id": f"{project_id}-action-{i}",
            }
        )

    # One doc per open question
    for i, q in enumerate(answer.open_questions):
        docs.append(
            {
                "text": f"Open question from meeting '{answer.title}' ({date}): {q.question} (Asked by: {q.asked_by})",
                "metadata": {
                    "project_id": project_id,
                    "date": date,
                    "type": "open_question",
                },
                "id": f"{project_id}-question-{i}",
            }
        )

    # Risk doc
    if answer.risks:
        docs.append(
            {
                "text": f"Risk noted in meeting '{answer.title}' ({date}): {answer.risks}",
                "metadata": {"project_id": project_id, "date": date, "type": "risk"},
                "id": f"{project_id}-risk",
            }
        )
    return docs


# function for embedding summary, action items, risks and open questions of a meeting
def embed_doc(docs: list[dict]):
    for doc in docs:
        result = emb_client.models.embed_content(
            model="gemini-embedding-2-preview",
            contents=doc["text"],
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT"),
        )
        doc["embedding"] = result.embeddings[0].values
    return docs


# function to store the document in chroma db
def store(docs: list[dict]):
    docs = embed_doc(docs)
    collection.add(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        embeddings=[d["embedding"] for d in docs],
        metadatas=[d["metadata"] for d in docs],
    )


# function to embed user query
def embed_query(query: str):
    result = emb_client.models.embed_content(
        model="gemini-embedding-2-preview",
        contents=query,
        config=types.EmbedContentConfig(task_type="RETRIEVAL_QUERY"),
    )
    return result.embeddings[0].values


@tool
def search(query: str, project_id: str):
    """ "function to search for data related to user's query and return the related data which has the same project id as passed in arguments
    Args:
        query (str): The user query that needs to be answered
        project_id (str): The project id which belongs to the project user is asking questions about
    Returns:
        data (List[List[Documents]]): List of documents semantically similar to user's query
    """
    result = embed_query(query)
    result = collection.query(query_embeddings=result, where={"project_id": project_id})
    data = result["documents"]
    return data
