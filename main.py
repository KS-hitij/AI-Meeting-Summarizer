from fastapi import FastAPI, UploadFile, File
import os
import tempfile
import shutil
import redis
from tasks import summarize_file


REDIS_HOST = os.getenv('REDIS_HOST')
REDIS_PASSWORD = os.getenv('REDIS_PASSWORD')
REDIS_USERNAME= os.getenv('REDIS_USERNAME')
REDIS_PORT= os.getenv('REDIS_PORT')

app = FastAPI()
r = redis.Redis(
    host=REDIS_HOST,
    port=REDIS_PORT,
    decode_responses=True,
    username=REDIS_USERNAME,
    password=REDIS_PASSWORD
)

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@app.post("/summarize")
async def summarize(file: UploadFile = File(...)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name
    if file.content_type == 'audio/mpeg' or file.content_type=='audio/wav' or file.content_type== 'audio/mp4':
        file_type = "audio"
    else:
        file_type = "text"
    task = summarize_file.delay(tmp_path,file.filename,file_type)
    return {"task_id":task.id}


@app.get('/tasks/{task_id}')
def get_task_status(task_id:str):
    async_result = summarize_file.AsyncResult(task_id)
    return{
        "task_id":task_id,
        "status":async_result.status,
        "result":async_result.result if async_result.ready() else None
    }