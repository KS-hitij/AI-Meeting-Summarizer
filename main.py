from fastapi import Depends, FastAPI, UploadFile, File, HTTPException, status
import os
import tempfile
import logging
from db.project import Project
from db.user import User
from sqlalchemy import select
from datetime import datetime
from tasks import summarize_file, rag
from pydantic import BaseModel, EmailStr, StringConstraints
from typing import Annotated, Any
from contextlib import asynccontextmanager
from db.database import init_db, get_db
from sqlalchemy.ext.asyncio import AsyncSession
from redis_client import r
import bcrypt
import uuid
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from util import (
    AUTH_TOKEN_EXPIRE_MINUTES,
    create_access_token,
    register_exception_handlers,
    verify_auth_token,
)

logger = logging.getLogger(__name__)


class RagRequestClass(BaseModel):
    query: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)
    ]
    project_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ]


class ProjectCreateClass(BaseModel):
    project_id: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=128)
    ]
    project_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)
    ]


class LoginClass(BaseModel):
    user_email: EmailStr
    password: Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class UserCreateClass(BaseModel):
    user_name: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)
    ]
    user_email: EmailStr
    password: Annotated[str, StringConstraints(min_length=1, max_length=1024)]


class TokenResponse(BaseModel):
    message: str
    token: str


class MessageResponse(BaseModel):
    message: str


class TaskSubmissionResponse(BaseModel):
    task_id: str


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    # Celery task results differ by task type and may be absent until completion.
    result: Any | None = None


class ProjectResponse(BaseModel):
    project_id: str
    project_name: str


class ProjectListResponse(BaseModel):
    projects: list[ProjectResponse]


class APIErrorBody(BaseModel):
    code: str
    message: str
    details: Any | None = None


class APIErrorResponse(BaseModel):
    error: APIErrorBody


API_ERROR_RESPONSES = {
    code: {"model": APIErrorResponse}
    for code in (
        status.HTTP_400_BAD_REQUEST,
        status.HTTP_401_UNAUTHORIZED,
        status.HTTP_403_FORBIDDEN,
        status.HTTP_404_NOT_FOUND,
        status.HTTP_409_CONFLICT,
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        status.HTTP_429_TOO_MANY_REQUESTS,
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
}


UPLOAD_DIRECTORY = "uploads/"
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)
auth_scheme = HTTPBearer(auto_error=False)
TASK_METADATA_TTL_SECONDS = AUTH_TOKEN_EXPIRE_MINUTES * 60
MAX_UPLOAD_SIZE_BYTES = int(os.getenv("MAX_UPLOAD_SIZE_BYTES", str(25 * 1024 * 1024)))
UPLOAD_CHUNK_SIZE_BYTES = 64 * 1024


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(auth_scheme),
    db: AsyncSession = Depends(get_db),
):
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required"
        )
    payload = verify_auth_token(credentials.credentials)
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
register_exception_handlers(app)


@app.post(
    "/signup",
    response_model=TokenResponse,
    responses=API_ERROR_RESPONSES,
)
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
    token = create_access_token(
        {"user_id": user_data.user_id, "user_email": user_data.user_email}
    )
    return {"message": "User created successfully", "token": token}


@app.post(
    "/login",
    response_model=TokenResponse,
    responses=API_ERROR_RESPONSES,
)
async def login(user: LoginClass, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(User).filter(User.user_email == user.user_email))
    db_user = result.scalars().first()
    if not db_user or not bcrypt.checkpw(
        user.password.encode("utf-8"), db_user.password_hash.encode("utf-8")
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    token = create_access_token(
        {"user_id": db_user.user_id, "user_email": db_user.user_email}
    )
    return {"message": "Login successful", "token": token}


@app.post(
    "/summarize/{project_id}",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=TaskSubmissionResponse,
    responses=API_ERROR_RESPONSES,
)
async def summarize(
    project_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    tmp_path = None
    task_metadata_key = None
    try:
        original_filename = os.path.basename((file.filename or "").replace("\\", "/"))
        if not original_filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="A filename is required",
            )

        media_type = (file.content_type or "").split(";", 1)[0].strip().lower()
        supported_types = {
            "audio/mpeg": ("audio", ".mp3"),
            "audio/wav": ("audio", ".wav"),
            "audio/x-wav": ("audio", ".wav"),
            "audio/mp4": ("audio", ".m4a"),
            "text/plain": ("text", ".txt"),
        }
        if media_type not in supported_types:
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail="Only MPEG, WAV, MP4 audio, and plain text files are supported",
            )
        file_type, suffix = supported_types[media_type]

        if file.size is not None and file.size > MAX_UPLOAD_SIZE_BYTES:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"File exceeds the {MAX_UPLOAD_SIZE_BYTES}-byte upload limit",
            )

        await get_project_or_403(project_id, current_user, db)

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp_path = tmp.name
            total_size = 0
            while chunk := await file.read(UPLOAD_CHUNK_SIZE_BYTES):
                total_size += len(chunk)
                if total_size > MAX_UPLOAD_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds the {MAX_UPLOAD_SIZE_BYTES}-byte upload limit",
                    )
                tmp.write(chunk)

        if total_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="The uploaded file is empty",
            )

        meeting_id = uuid.uuid4().hex
        task_id = uuid.uuid4().hex
        task_metadata_key = f"task:{task_id}"
        r.hset(
            task_metadata_key,
            mapping={
                "user_id": current_user.user_id,
                "project_id": project_id,
            },
        )
        r.expire(task_metadata_key, TASK_METADATA_TTL_SECONDS)
        summarize_file.apply_async(
            args=[tmp_path, original_filename, file_type, project_id, meeting_id],
            task_id=task_id,
        )

        # The Celery worker owns the temporary file after the task is queued.
        tmp_path = None
        return {"task_id": task_id}
    except Exception:
        if task_metadata_key:
            try:
                r.delete(task_metadata_key)
            except Exception:
                logger.exception("Failed to remove task metadata after upload failure")
        if tmp_path:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            except OSError:
                logger.exception("Failed to remove temporary upload file")
        raise
    finally:
        await file.close()


@app.get(
    "/tasks/{task_id}",
    response_model=TaskStatusResponse,
    responses=API_ERROR_RESPONSES,
)
async def get_task_status(
    task_id: str,
    current_user: User = Depends(get_current_user),
):
    task_metadata = r.hgetall(f"task:{task_id}")
    if not task_metadata or task_metadata.get("user_id") != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found",
        )

    async_result = summarize_file.AsyncResult(task_id)
    return {
        "task_id": task_id,
        "status": async_result.status,
        "result": async_result.result if async_result.ready() else None,
    }


@app.post(
    "/rag_query",
    response_model=TaskSubmissionResponse,
    responses=API_ERROR_RESPONSES,
)
async def rag_query(
    rag_req: RagRequestClass,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    await get_project_or_403(rag_req.project_id, current_user, db)
    task = rag.delay(rag_req.query, rag_req.project_id)
    return {"task_id": task.id}


@app.post(
    "/projects",
    response_model=MessageResponse,
    responses=API_ERROR_RESPONSES,
)
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


@app.get(
    "/projects",
    response_model=ProjectListResponse,
    responses=API_ERROR_RESPONSES,
)
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
