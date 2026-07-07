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

class Answer(BaseModel):
    title:str
    summary:str
    action_items:list[ActionItem]
    open_questions:list[OpenQuestion]
    risks: Optional[Risk]



tools =[transcribe]
tools_by_name={tool.name: tool for tool in tools}

model = model.bind_tools(tools)
model_with_structured_output = model.with_structured_output(Answer)