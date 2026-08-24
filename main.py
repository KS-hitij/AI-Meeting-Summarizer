from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, status
import os
import tempfile
import shutil
from db.project import Project
from db.user import User
from sqlalchemy import select
from datetime import datetime
from tasks import summarize_file, rag
from pydantic import BaseModel
from contextlib import asynccontextmanager
from db.database import init_db, get_db
from sqlalchemy.ext.asyncio import AsyncSession
from redis_client import r
import bcrypt
import uuid
import base64
import hashlib
import hmac
import json
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer


class RagRequestClass(BaseModel):
    query: str
    project_id: str


class ProjectCreateClass(BaseModel):
    project_id: str
    project_name: str


class LoginClass(BaseModel):
    user_email: str
    password: str


class UserCreateClass(BaseModel):
    user_name: str
    user_email: str
    password: str


UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
auth_scheme = HTTPBearer(auto_error=False)
AUTH_SECRET = os.getenv("AUTH_SECRET", "change-me-in-production")


def _sign_auth_payload(payload: dict) -> str:
    raw_payload = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode(
        "utf-8"
    )
    payload_b64 = base64.urlsafe_b64encode(raw_payload).decode("utf-8").rstrip("=")
    signature = hmac.new(
        AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
    ).digest()
    signature_b64 = base64.urlsafe_b64encode(signature).decode("utf-8").rstrip("=")
    return f"{payload_b64}.{signature_b64}"


def _verify_auth_token(token: str) -> dict:
    try:
        payload_b64, signature_b64 = token.split(".")
        expected_signature = hmac.new(
            AUTH_SECRET.encode("utf-8"), payload_b64.encode("utf-8"), hashlib.sha256
        ).digest()
        expected_signature_b64 = (
            base64.urlsafe_b64encode(expected_signature).decode("utf-8").rstrip("=")
        )
        if not hmac.compare_digest(signature_b64, expected_signature_b64):
            raise ValueError("Invalid signature")
        padded_payload = payload_b64 + "=" * (-len(payload_b64) % 4)
        return json.loads(base64.urlsafe_b64decode(padded_payload.encode("utf-8")))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token"
        ) from exc


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    db: AsyncSession = Depends(get_db),
):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    payload = _verify_auth_token(credentials.credentials)
    user_id = payload.get("user_id")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload"
        )
    result = await db.execute(select(User).filter(User.user_id == user_id))
    user = result.scalars().first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found"
        )
    return user


async def get_project_or_403(project_id: str, current_user: User, db: AsyncSession):
    result = await db.execute(select(Project).filter(Project.project_id == project_id))
    project = result.scalars().first()
    if not project:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
        )
    project_members = [
        member.strip()
        for member in (project.project_members or "").split(",")
        if member.strip()
    ]
    if current_user.user_id not in project_members:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this project",
        )
    return project


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/signup")
async def signup(user: UserCreateClass, db: AsyncSession = Depends(get_db)):
    password_hash = bcrypt.hashpw(user.password.encode("utf-8"), bcrypt.gensalt())
    user_data = User(
        user_id=uuid.uuid4().hex,
        user_name=user.user_name,
        user_email=user.user_email,
        password_hash=password_hash.decode("utf-8"),
        created_at=datetime.now(),
    )
    db.add(user_data)
    await db.commit()
    token = _sign_auth_payload(
        {"user_id": user_data.user_id, "user_email": user_data.user_email}
    )
    return {"message": "User created successfully", "token": token}


@app.post("/login")
async def login(user: LoginClass, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.user_email == user.user_email))
    db_user = result.scalars().first()
    if not db_user or not bcrypt.checkpw(
        user.password.encode("utf-8"), db_user.password_hash.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = _sign_auth_payload(
        {"user_id": db_user.user_id, "user_email": db_user.user_email}
    )
    return {"message": "Login successful", "token": token}


@app.post("/summarize/{project_id}")
async def summarize(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    suffix = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    if file.content_type in ("audio/mpeg", "audio/wav", "audio/mp4"):
        file_type = "audio"
    elif file.content_type == "text/plain":
        file_type = "text"
    else:
        raise ValueError("Wrong type of file passed only text and audio allowed")

    await get_project_or_403(project_id, current_user, db)

    task = summarize_file.delay(tmp_path, file.filename, file_type, project_id)
    return {"task_id": task.id}


@app.get("/tasks/{task_id}")
def get_task_status(task_id: str):
    async_result = summarize_file.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": async_result.status,
        "result": async_result.result if async_result.ready() else None,
    }


@app.post("/rag_query")
async def rag_query(
    rag_req: RagRequestClass,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await get_project_or_403(rag_req.project_id, current_user, db)
    task = rag.delay(rag_req.query, rag_req.project_id)
    return {"task_id": task.id}


@app.post("/projects")
async def create_project(
    project: ProjectCreateClass,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    project = Project(
        project_id=project.project_id,
        project_name=project.project_name,
        created_at=datetime.now(),
        updated_at=datetime.now(),
        project_members=current_user.user_id,
    )
    db.add(project)
    await db.commit()
    return {"message": "Project created successfully"}


@app.get("/projects")
async def get_projects(
    db: AsyncSession = Depends(get_db), current_user: User = Depends(get_current_user)
):
    result = await db.execute(select(Project))
    projects = result.scalars().all()
    allowed_projects = []
    for p in projects:
        project_members = [
            member.strip()
            for member in (p.project_members or "").split(",")
            if member.strip()
        ]
        if current_user.user_id in project_members:
            allowed_projects.append(p)
    return {
        "projects": [
            {"project_id": p.project_id, "project_name": p.project_name}
            for p in allowed_projects
        ]
    }
