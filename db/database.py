import os
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine,async_sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_async_engine(DATABASE_URL)

AsyncSessionLocal = async_sessionmaker(engine,expire_on_commit=False)

async def get_db() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session
