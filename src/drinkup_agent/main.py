"""Main application entry point."""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import logging
import json
from contextlib import asynccontextmanager

from .config import settings
from .api.chat_stream import router as chat_stream_router
from .api.insights import router as insights_router
from .database.init import init_database

# 配置日志级别
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),  # 输出到控制台
    ],
)

# 获取日志记录器
logger = logging.getLogger(__name__)


async def log_requests_middleware(request: Request, call_next):
    """记录所有请求的中间件，特别关注chat相关的请求。"""
    # 记录请求基本信息
    logger.info(f"Incoming request: {request.method} {request.url.path}")

    # 如果是流式响应端点，直接处理，避免读取body的问题
    if request.url.path.endswith("/stream"):
        response = await call_next(request)
        return response

    # 如果是我们关心的聊天端点，记录更详细的信息
    if "/chat" in request.url.path:
        logger.info("Chat request details:")
        logger.info(f"- Method: {request.method}")
        logger.info(f"- URL: {request.url}")
        logger.info(f"- Headers: {dict(request.headers)}")
        logger.info(f"- Query params: {dict(request.query_params)}")

        # 如果是POST请求，尝试读取请求体
        if request.method == "POST":
            try:
                # 读取原始请求体
                body = await request.body()
                if body:
                    try:
                        # 尝试解析为JSON
                        body_json = json.loads(body.decode("utf-8"))
                        logger.info(
                            f"- Request body (JSON): {json.dumps(body_json, indent=2, ensure_ascii=False)}"
                        )
                    except json.JSONDecodeError:
                        logger.info(
                            f"- Request body (raw): {body.decode('utf-8', errors='ignore')}"
                        )
                else:
                    logger.info("- Request body: Empty")

                # 重新构造请求，因为body已经被读取了
                async def receive():
                    return {"type": "http.request", "body": body}

                request._receive = receive

            except Exception as e:
                logger.error(f"Error reading request body: {e}")

    # 处理请求
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.error(f"Error processing request: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        raise


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    # Startup
    await init_database()
    yield
    # Shutdown
    print("Shutting down DrinkUp Agent...")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="DrinkUp Agent",
        description="AI agent service for DrinkUp chatbot interactions",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure properly in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add request logging middleware
    app.middleware("http")(log_requests_middleware)

    # Include routers
    app.include_router(chat_stream_router, prefix=f"{settings.api_prefix}")
    app.include_router(insights_router, prefix=f"{settings.api_prefix}")

    @app.get("/")
    async def root():
        """Root endpoint."""
        return {"name": "DrinkUp Agent", "version": "0.1.0", "status": "running"}

    @app.get("/health")
    async def health():
        """Health check endpoint."""
        return {"status": "healthy"}

    return app


app = create_app()


if __name__ == "__main__":
    uvicorn.run(
        "drinkup_agent.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=settings.debug,
    )
