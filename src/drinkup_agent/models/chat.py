"""Chat-related models."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Individual chat message."""

    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., description="Message content")


class ImageAttachment(BaseModel):
    """Image attachment for chat."""

    image_base64: Optional[str] = Field(None, description="Base64 encoded image data")
    mime_type: Optional[str] = Field("image/jpeg", description="MIME type of the image")
    
    class Config:
        populate_by_name = True


class ChatParams(BaseModel):
    """Parameters for chat context."""

    user_stock: str = Field(default="", description="User's current stock/inventory")
    user_info: str = Field(default="", description="User information/preferences")
    image_attachment_list: Optional[List[ImageAttachment]] = Field(
        None, description="List of image attachments"
    )


class ChatV2Request(BaseModel):
    """Request for chat v2 endpoint."""

    user_message: str = Field(..., description="User's message")
    user_id: str = Field(..., description="User ID for memory management")
    conversation_id: Optional[str] = Field(
        None, description="Conversation ID for context"
    )
    history: Optional[List[ChatMessage]] = Field(
        None,
        description=(
            "Client-provided prior turns. Used to re-seed context when the "
            "server-side conversation cache (24h Redis) has expired."
        ),
    )
    params: ChatParams = Field(
        default_factory=ChatParams, description="Chat parameters"
    )


class ChatV2Response(BaseModel):
    """Response from chat v2 endpoint."""

    conversation_id: str = Field(..., description="Conversation ID")
    content: str = Field(..., description="AI response content")
    usage: Optional[Dict[str, Any]] = Field(None, description="Token usage information")
