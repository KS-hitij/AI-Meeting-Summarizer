import os
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker
from .base import Base

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)

AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db():
    async with AsyncSessionLocal() as session:
        yield session


async def init_db():
    print("Initializing database")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
