from pydantic import BaseSettings


class LLMClientConfig(BaseSettings):
    host: str
    port: int


class RetrieverEncoderConfig(BaseSettings):
    model_name: str


class RetrieverConfig(BaseSettings):
    collection_name: str
    qdrant_host: str
    qdrant_port: int
    top_k: int


class LoggingConfig(BaseSettings):
    log_path: str


class Config(BaseSettings):
    objective: str
    initial_task: str
    llm_client: LLMClientConfig
    retriever_encoder: RetrieverEncoderConfig
    retriever: RetrieverConfig
    logger: LoggingConfig

    class Config:
        env_file = ".env"
