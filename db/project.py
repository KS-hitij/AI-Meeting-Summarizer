from sqlalchemy import Column, Integer, String, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from datetime import datetime

from .base import Base


class Project(Base):
    __tablename__ = "projects"
    project_id: Mapped[str] = mapped_column(String, primary_key=True)
    project_name: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow(), onupdate=datetime.utcnow()
    )
