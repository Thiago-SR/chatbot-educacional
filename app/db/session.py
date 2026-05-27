from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings


def get_engine():
    """Retorna o engine assíncrono para o banco de dados."""
    return create_async_engine(settings.DATABASE_URL, echo=False)


engine = get_engine()
async_session_factory = async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncSession:
    """Fornece uma sessão assíncrona configurada para injeção de dependência."""
    async with async_session_factory() as session:
        yield session
