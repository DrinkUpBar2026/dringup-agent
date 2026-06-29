"""Configuration settings for DrinkUp Agent."""

from pydantic_settings import BaseSettings
from typing import Optional

# Load .env with override=True to ensure .env file values take precedence over environment variables
try:
    from dotenv import load_dotenv

    load_dotenv(override=True)
except Exception:
    # If python-dotenv is not available, pydantic-settings will still read
    # values from the .env file for this Settings model.
    pass


class Settings(BaseSettings):
    """Application settings loaded from .env file with priority over environment variables."""

    # OpenAI Configuration
    openai_api_key: str
    openai_model: str = "gpt-4o-mini"
    openai_base_url: Optional[str] = None
    # 「开喝时刻」场景总结用的模型；留空则复用 openai_model。可设成更便宜的模型省钱。
    insights_model: Optional[str] = None

    # LangSmith / LangChain Tracing Configuration
    # When enabled, LangChain will send traces to LangSmith
    langsmith_tracing: bool = False
    langsmith_endpoint: Optional[str] = "https://api.smith.langchain.com"
    langsmith_api_key: Optional[str] = None
    langsmith_project: Optional[str] = "drinkup-dev"

    # Mem0 Configuration (Optional - enables memory tools when provided)
    memory_enabled: bool = False
    mem0_api_key: Optional[str] = None
    mem0_base_url: Optional[str] = "https://api.mem0.ai"

    # Mem0 Embedding API Configuration (separate from main OpenAI config)
    mem0_embedding_api_key: Optional[str] = None
    mem0_embedding_base_url: Optional[str] = None
    mem0_embedding_model: str = "text-embedding-3-large"

    # Milvus Configuration for Mem0 Vector Store
    milvus_url: str = "http://localhost:19530"  # Milvus server URL
    milvus_token: Optional[str] = (
        None  # Authentication token (optional for local setup)
    )
    milvus_collection_name: str = "mem0"  # Name of the Milvus collection
    milvus_db_name: str = ""  # Database name (empty for default)

    # Memgraph Configuration for Mem0 Graph Store (deprecated, use Milvus instead)
    memgraph_url: str = "bolt://localhost:7687"
    memgraph_username: str = "memgraph"
    memgraph_password: str = ""

    # OpenAI Embedding Configuration for Mem0 (deprecated, use mem0_embedding_* instead)
    embedding_model: str = "text-embedding-3-large"
    embedding_dims: int = 1536  # Dimensions for embedding model

    # Server Configuration
    server_host: str = "0.0.0.0"
    server_port: int = 8001

    # DrinkUp Backend Configuration
    drinkup_backend_url: str = "http://localhost:8080"
    drinkup_backend_timeout: float = 600.0

    # Redis Configuration
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_username: Optional[str] = None
    redis_password: Optional[str] = None

    # MySQL Configuration
    mysql_host: str = "localhost"
    mysql_port: int = 3306
    mysql_database: str = "drinkup"
    mysql_user: str = "root"
    mysql_password: str = ""

    # API Configuration
    api_prefix: str = "/api"
    debug: bool = False

    class Config:
        # Priority: .env file values override environment variables
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


settings = Settings()
