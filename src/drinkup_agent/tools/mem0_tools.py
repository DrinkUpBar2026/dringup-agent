"""Mem0 tools for LangChain agent."""

from typing import Optional, Type, Any, Dict, List
from langchain.tools import BaseTool
from langchain.callbacks.manager import (
    CallbackManagerForToolRun,
    AsyncCallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
from mem0 import Memory
import json
import logging

logger = logging.getLogger(__name__)


class Mem0Manager:
    """Manager class for Mem0 operations."""

    # Class-level cache for Memory instances
    _memory_instance = None
    _initialized_config = None

    def __init__(
        self,
        api_key: Optional[str] = None,
        vector_store_config: Optional[Dict[str, Any]] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
    ):
        """Initialize Mem0 manager with optional vector store configuration (Milvus or Memgraph)."""
        self.enabled = bool(api_key or embedding_api_key)  # Enable if either key is provided
        if self.enabled:
            try:
                # Set the API keys in environment for Mem0
                import os

                # Use embedding-specific API key if provided, otherwise fall back to general API key
                effective_api_key = embedding_api_key or api_key
                if effective_api_key:
                    os.environ["OPENAI_API_KEY"] = effective_api_key

                # Set base URL for embeddings if provided
                if embedding_base_url:
                    os.environ["OPENAI_BASE_URL"] = embedding_base_url
                elif "OPENAI_API_BASE" in os.environ:
                    os.environ["OPENAI_BASE_URL"] = os.environ.pop("OPENAI_API_BASE")

                # Check if we need to create a new Memory instance
                config_key = (
                    str(vector_store_config) if vector_store_config else "default"
                )

                if (
                    Mem0Manager._memory_instance is None
                    or Mem0Manager._initialized_config != config_key
                ):
                    # Initialize Mem0 with vector store configuration if provided
                    if vector_store_config:
                        store_type = vector_store_config.get("type", "milvus")

                        if store_type == "milvus":
                            # Milvus configuration
                            config = {
                                "vector_store": {
                                    "provider": "milvus",
                                    "config": {
                                        "url": vector_store_config["url"],
                                        "collection_name": vector_store_config.get(
                                            "collection_name", "mem0"
                                        ),
                                        "embedding_model_dims": str(
                                            vector_store_config.get(
                                                "embedding_dims", 1536
                                            )
                                        ),
                                        "db_name": vector_store_config.get(
                                            "db_name", ""
                                        ),
                                    },
                                },
                                "embedder": {
                                    "provider": "openai",
                                    "config": {
                                        "model": vector_store_config.get(
                                            "embedding_model", "text-embedding-3-large"
                                        ),
                                        "api_key": embedding_api_key or api_key,  # Use embedding API key if available
                                        "openai_base_url": embedding_base_url,
                                    },
                                },
                                # mem0's fact-extraction LLM. Must match the embedding
                                # provider (both routed through embedding creds/base_url).
                                # Aliyun Bailian etc. reject OpenAI model names, so make
                                # the model explicit/overridable via MEM0_LLM_MODEL.
                                "llm": {
                                    "provider": "openai",
                                    "config": {
                                        "model": os.getenv("MEM0_LLM_MODEL", "qwen-plus"),
                                        "api_key": embedding_api_key or api_key,
                                        "openai_base_url": embedding_base_url,
                                    },
                                },
                                "version": "v1.1",
                            }
                            # Add token if provided
                            if vector_store_config.get("token"):
                                config["vector_store"]["config"]["token"] = (
                                    vector_store_config["token"]
                                )

                            Mem0Manager._memory_instance = Memory.from_config(
                                config_dict=config
                            )
                            logger.info("Mem0 initialized with Milvus vector store")

                        elif store_type == "memgraph":
                            # Legacy Memgraph configuration
                            config = {
                                "graph_store": {
                                    "provider": "memgraph",
                                    "config": {
                                        "url": vector_store_config["url"],
                                        "username": vector_store_config["username"],
                                        "password": vector_store_config["password"],
                                    },
                                },
                                "embedder": {
                                    "provider": "openai",
                                    "config": {
                                        "model": vector_store_config.get(
                                            "embedding_model", "text-embedding-3-large"
                                        ),
                                        "api_key": embedding_api_key or api_key,  # Use embedding API key if available
                                        "embedding_dims": 1536,
                                    },
                                },
                                "version": "v1.1",
                            }
                            Mem0Manager._memory_instance = Memory.from_config(
                                config_dict=config
                            )
                            logger.info("Mem0 initialized with Memgraph graph store")
                    else:
                        # Default configuration without vector store
                        Mem0Manager._memory_instance = Memory()
                        logger.info("Mem0 initialized with default configuration")

                    Mem0Manager._initialized_config = config_key

                # Use the cached instance
                self.memory = Mem0Manager._memory_instance
            except Exception as e:
                logger.error("Failed to initialize Mem0: %s", str(e))
                self.enabled = False
                self.memory = None
        else:
            self.memory = None
            logger.info("Mem0 not configured")

    def is_enabled(self) -> bool:
        """Check if Mem0 is enabled."""
        return self.enabled and self.memory is not None


class AddMemoryInput(BaseModel):
    """Input for adding memory."""

    memory_content: str = Field(
        description="The information to remember about the user"
    )


class AddMemoryTool(BaseTool):
    """Tool for adding memories to Mem0."""

    name: str = "add_memory"
    description: str = """Store important information about the user for future reference. 
    Use this when the user shares preferences, personal information, instructions, or any details worth remembering."""
    args_schema: Type[BaseModel] = AddMemoryInput

    mem0_manager: Mem0Manager
    user_id: str

    def _run(
        self,
        memory_content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous version (not implemented, use async)."""
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        memory_content: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Add a memory asynchronously."""
        if not self.mem0_manager.is_enabled():
            return "Memory service is not available"

        try:
            result = self.mem0_manager.memory.add(
                messages=[{"role": "user", "content": memory_content}],
                user_id=self.user_id,
            )

            # Return raw result from Mem0
            return json.dumps(result, ensure_ascii=False) if result else "{}"

        except Exception as e:
            logger.error("Error adding memory: %s", str(e))
            return f"Error storing memory: {str(e)}"


class SearchMemoryInput(BaseModel):
    """Input for searching memories."""

    query: str = Field(description="Search query to find relevant memories")
    limit: int = Field(default=5, description="Maximum number of memories to retrieve")


class SearchMemoryTool(BaseTool):
    """Tool for searching memories in Mem0."""

    name: str = "search_memory"
    description: str = """Search for relevant memories about the user to provide personalized responses.
    Use this to recall user preferences, past conversations, or any stored information."""
    args_schema: Type[BaseModel] = SearchMemoryInput

    mem0_manager: Mem0Manager
    user_id: str

    def _run(
        self,
        query: str,
        limit: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous version (not implemented, use async)."""
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        query: str,
        limit: int = 5,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Search memories asynchronously."""
        if not self.mem0_manager.is_enabled():
            return "Memory service is not available"

        try:
            results = self.mem0_manager.memory.search(
                query=query, user_id=self.user_id, limit=limit
            )

            # Return raw result from Mem0
            return json.dumps(results, ensure_ascii=False) if results else "{}"

        except Exception as e:
            logger.error("Error searching memories: %s", str(e))
            return f"Error searching memories: {str(e)}"


class GetAllMemoriesInput(BaseModel):
    """Input for getting all memories."""

    limit: int = Field(default=20, description="Maximum number of memories to retrieve")


class GetAllMemoriesTool(BaseTool):
    """Tool for getting all user memories from Mem0."""

    name: str = "get_all_memories"
    description: str = """Retrieve all stored memories about the user.
    Use this to get a complete overview of what you know about the user."""
    args_schema: Type[BaseModel] = GetAllMemoriesInput

    mem0_manager: Mem0Manager
    user_id: str

    def _run(
        self, limit: int = 20, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Synchronous version (not implemented, use async)."""
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        limit: int = 20,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Get all memories asynchronously."""
        if not self.mem0_manager.is_enabled():
            return "Memory service is not available"

        try:
            results = self.mem0_manager.memory.get_all(
                user_id=self.user_id, limit=limit
            )

            # Return raw result from Mem0
            return json.dumps(results, ensure_ascii=False) if results else "{}"

        except Exception as e:
            logger.error("Error getting all memories: %s", str(e))
            return f"Error retrieving memories: {str(e)}"


class UpdateMemoryInput(BaseModel):
    """Input for updating memory."""

    memory_id: str = Field(description="ID of the memory to update")
    new_content: str = Field(description="Updated memory content")


class UpdateMemoryTool(BaseTool):
    """Tool for updating memories in Mem0."""

    name: str = "update_memory"
    description: str = """Update an existing memory with new or corrected information.
    Use this when you need to modify outdated or incorrect stored information."""
    args_schema: Type[BaseModel] = UpdateMemoryInput

    mem0_manager: Mem0Manager
    user_id: str

    def _run(
        self,
        memory_id: str,
        new_content: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """Synchronous version (not implemented, use async)."""
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        memory_id: str,
        new_content: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Update a memory asynchronously."""
        if not self.mem0_manager.is_enabled():
            return "Memory service is not available"

        try:
            result = self.mem0_manager.memory.update(
                memory_id=memory_id, data=new_content
            )
            return (
                json.dumps(result, ensure_ascii=False)
                if result
                else '{"status": "updated"}'
            )

        except Exception as e:
            logger.error("Error updating memory: %s", str(e))
            return f"Error updating memory: {str(e)}"


class DeleteMemoryInput(BaseModel):
    """Input for deleting memory."""

    memory_id: str = Field(description="ID of the memory to delete")


class DeleteMemoryTool(BaseTool):
    """Tool for deleting memories from Mem0."""

    name: str = "delete_memory"
    description: str = """Delete a specific memory that is no longer relevant or accurate.
    Use this to remove outdated or incorrect information."""
    args_schema: Type[BaseModel] = DeleteMemoryInput

    mem0_manager: Mem0Manager
    user_id: str

    def _run(
        self, memory_id: str, run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> str:
        """Synchronous version (not implemented, use async)."""
        raise NotImplementedError("Use async version")

    async def _arun(
        self,
        memory_id: str,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None,
    ) -> str:
        """Delete a memory asynchronously."""
        if not self.mem0_manager.is_enabled():
            return "Memory service is not available"

        try:
            result = self.mem0_manager.memory.delete(memory_id=memory_id)
            return (
                json.dumps(result, ensure_ascii=False)
                if result
                else '{"status": "deleted"}'
            )

        except Exception as e:
            logger.error("Error deleting memory: %s", str(e))
            return f"Error deleting memory: {str(e)}"


def create_mem0_tools(api_key: Optional[str], user_id: str) -> List[BaseTool]:
    """
    Create all Mem0 tools for a specific user.

    Args:
        api_key: Mem0 API key (this should be the OpenAI API key)
        user_id: User ID for memory operations

    Returns:
        List of configured Mem0 tools
    """
    # Import settings here to get vector store configuration
    from ..config import settings

    # Prepare vector store configuration
    vector_store_config = None

    # Use Milvus as the primary vector store
    use_milvus = True  # Enable Milvus for persistent memory storage

    # Use mem0-specific embedding settings if available, otherwise fall back to general settings
    embedding_model = settings.mem0_embedding_model or settings.embedding_model
    embedding_api_key = settings.mem0_embedding_api_key
    embedding_base_url = settings.mem0_embedding_base_url

    if use_milvus:
        vector_store_config = {
            "type": "milvus",
            "url": settings.milvus_url,
            "token": settings.milvus_token,
            "collection_name": settings.milvus_collection_name,
            "db_name": settings.milvus_db_name,
            "embedding_model": embedding_model,
            "embedding_dims": settings.embedding_dims,
        }
        logger.info("Using Milvus vector store for Mem0")
    elif settings.memgraph_password:  # Fallback to Memgraph if password is configured
        vector_store_config = {
            "type": "memgraph",
            "url": settings.memgraph_url,
            "username": settings.memgraph_username,
            "password": settings.memgraph_password,
            "embedding_model": embedding_model,
        }
        logger.info("Using Memgraph configuration for Mem0")
    else:
        logger.info("Using default Mem0 storage (SQLite + ChromaDB)")

    # Create shared Mem0 manager with vector store config and embedding API settings
    manager = Mem0Manager(
        api_key,
        vector_store_config,
        embedding_api_key=embedding_api_key,
        embedding_base_url=embedding_base_url
    )

    # Create and configure tools - they inherit name and description from class attributes
    tools = []

    if manager.is_enabled():
        tools = [
            AddMemoryTool(mem0_manager=manager, user_id=user_id),
            SearchMemoryTool(mem0_manager=manager, user_id=user_id),
            GetAllMemoriesTool(mem0_manager=manager, user_id=user_id),
            UpdateMemoryTool(mem0_manager=manager, user_id=user_id),
            DeleteMemoryTool(mem0_manager=manager, user_id=user_id),
        ]

    return tools
