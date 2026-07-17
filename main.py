from fastapi import Depends, FastAPI, UploadFile, File
import os
import tempfile
import shutil
from db.project import Project
from sqlalchemy import select
from datetime import datetime
from tasks import summarize_file, rag
from pydantic import BaseModel
from contextlib import asynccontextmanager
from db.database import init_db, get_db
from sqlalchemy.ext.asyncio import AsyncSession
from redis_client import r


class RagRequestClass(BaseModel):
    query:str
    project_id:str

class ProjectCreateClass(BaseModel):
    project_id: str
    project_name: str

UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/summarize/{project_id}")
async def summarize(project_id: str, file: UploadFile = File(...), db: AsyncSession = Depends(get_db)):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    if file.content_type in ('audio/mpeg', 'audio/wav', 'audio/mp4'):
        file_type = "audio"
    elif file.content_type == 'text/plain':
        file_type = "text"
    else:
        raise ValueError('Wrong type of file passed only text and audio allowed')

    result = await db.execute(select(Project).filter(Project.project_id == project_id))
    project = result.scalars().first()
    if not project:
        raise ValueError("Project not found")

    task = summarize_file.delay(tmp_path, file.filename, file_type, project_id)
    return {"task_id": task.id}


@app.get('/tasks/{task_id}')
def get_task_status(task_id:str):
    async_result = summarize_file.AsyncResult(task_id)
    return{
        "task_id":task_id,
        "status":async_result.status,
        "result":async_result.result if async_result.ready() else None
    }

@app.post("/rag_query")
def rag_query(rag_req:RagRequestClass):
    task  = rag.delay(rag_req.query,rag_req.project_id)
    return {"task_id":task.id}

@app.post("/projects")
async def create_project(project: ProjectCreateClass, db: AsyncSession = Depends(get_db)):
    print("Creating project")
    project = Project(
        project_id=project.project_id,
        project_name=project.project_name,
        created_at=datetime.now(),
        updated_at=datetime.now()
    )
    db.add(project)
    await db.commit()
    return {"message": "Project created successfully"}

@app.get("/projects")
async def get_projects(db: AsyncSession = Depends(get_db)):
    print("Getting projects")
    result = await db.execute(select(Project))
    projects = result.scalars().all()
    return {"projects": [{"project_id": p.project_id, "project_name": p.project_name} for p in projects]}
