from langchain.tools import tool
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
from langchain.chat_models import init_chat_model
import os
import shutil
import whisper

load_dotenv()

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
api_key = os.getenv("GOOGLE_API_KEY")

model = init_chat_model(
    "gemini-3.1-flash-lite",
)

@tool
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

@tool
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

class Answer(BaseModel):
    title:str
    summary:str
    action_items:list[ActionItem]
    open_questions:list[OpenQuestion]
    risks: Optional[Risk]



tools =[upload,read,transcribe]
tools_by_name={tool.name: tool for tool in tools}

model = model.bind_tools(tools)