from fastapi import FastAPI, UploadFile, File
import os
from message import system_msg
from agent import summarizer_agent
import tempfile
import shutil

app = FastAPI()

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    result = summarizer_agent.invoke({
        "messages": [system_msg, {"role": "user", "content": f"Here is the file path: {tmp_path}"}]
    })
    os.remove(tmp_path)
    return {"summary": result['structured_response']}