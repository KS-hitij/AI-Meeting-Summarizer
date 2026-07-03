from langchain.agents import create_agent
import os
import shutil

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

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

class Answer():
    summary:str

summarizer_agent = create_agent(model="google_genai:gemini-3.1-flash-lite",tools=[upload,read])
