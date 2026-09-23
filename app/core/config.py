from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    hf_token: str
    qdrant_url: str = "http://192.168.29.146:6333"
    qdrant_collection: str = "documents"
    qdrant_password: Optional[str] = None
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    chat_model: str = "Qwen/Qwen2.5-72B-Instruct"
    chunk_size: int = 500
    chunk_overlap: int = 50
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    LANGFUSE_SECRET_KEY:str="sk-lf-592cede4-9288-4ecf-b7bc-f3becf9adf06"
    LANGFUSE_PUBLIC_KEY:str="pk-lf-ec9def0f-1e28-4b4e-ac06-cf49851e7981"
    LANGFUSE_BASE_URL:str="https://cloud.langfuse.com"

    class Config:
        env_file = ".env"


settings = Settings()
