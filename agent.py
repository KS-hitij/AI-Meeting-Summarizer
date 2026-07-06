from langchain.agents import create_agent
from pydantic import BaseModel
from dotenv import load_dotenv
import os
import shutil
import whisper

load_dotenv()

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
api_key = os.getenv("GOOGLE_API_KEY")

def upload(file:str)->str:
    """Upload a file to the server and return its name.
    Args:
        file (str): The path to the file to upload.
    Returns:
        file_name (str): The name of the uploaded file."""
    file_name = os.path.basename(file)
    destination = os.path.join(UPLOAD_DIRECTORY, file_name)
    shutil.copy(file, destination)
    return {"file_name": file_name}


def read(file_name:str)->str:
    """Read the content of a file given its name.
    Args:
        file_name (str): The name of the file to read.
    Returns:
        content (str): The content of the file."""
    file_path = os.path.join(UPLOAD_DIRECTORY, file_name)
    if not os.path.exists(file_path):
        return {"error": "File not found"}
    
    with open(file_path, "r") as file:
        content = file.read()
    return content

def trancribe(audio)->str:
    """Given the audio file transcribe it and return the transcription.
    Args:
        audio (): The audio file that needs to be transcribed
    Returns: 
        transcription (str): The transcription of the audio file"""
    model = whisper.load_model('turbo')
    result = model.transcribe(audio)
    transcription = result["text"]
    return transcription


class ActionItem(BaseModel):
    owner:str
    task:str

class Answer(BaseModel):
    title:str
    summary:str
    action_items:list[ActionItem]

summarizer_agent = create_agent(model="google_genai:gemini-3.1-flash-lite",tools=[upload,read],response_format=Answer)
