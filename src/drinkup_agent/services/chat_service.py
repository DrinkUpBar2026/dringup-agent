"""Chat service for handling AI conversations using LangChain."""

from ..models.chat import ChatV2Request, ChatV2Response
from .agent_service import AgentService
from ..repositories import PromptRepository


class ChatService:
    """Service for handling chat interactions using LangChain Agent."""

    def __init__(self):
        self.agent_service = AgentService()
        self.prompt_repository = PromptRepository()

    async def chat_v2(self, request: ChatV2Request) -> ChatV2Response:
        """Handle v2 chat request using LangChain agent with tools."""
        try:
            # Prepare image attachments
            image_attachments = None
            if request.params.image_attachment_list:
                image_attachments = [
                    {"image_base64": att.image_base64, "mime_type": att.mime_type}
                    for att in request.params.image_attachment_list
                    if att.image_base64
                ]
                if not image_attachments:
                    image_attachments = None

            # Call agent service
            result = await self.agent_service.chat_with_agent(
                user_message=request.user_message,
                user_id=request.user_id,
                conversation_id=request.conversation_id,
                user_stock=request.params.user_stock,
                user_info=request.params.user_info,
                image_attachments=image_attachments,
            )
        except Exception as e:
            import traceback

            print(f"Error in chat_v2: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            raise

        return ChatV2Response(
            conversation_id=result["conversation_id"],
            content=result["content"],
            usage=result["usage"],
        )
