"""Models for user insights (开喝时刻 / 场景总结)."""

from typing import List, Optional
from pydantic import BaseModel, Field

from .chat import ChatMessage


class MomentsRequest(BaseModel):
    """Request to summarize a user's drinking scenes from recent chat history."""

    user_id: str = Field(..., description="User ID")
    # 客户端把本地近 30 天聊天压缩后带来（剥图片、酒卡压成一句话）。
    history: List[ChatMessage] = Field(
        default_factory=list, description="Compressed recent chat turns [{role, content}]"
    )
    # 仅作展示口径提示（如 "本月"），不参与精确过滤。
    period_label: Optional[str] = Field(
        default="本月", description="Display label for the time window"
    )


class Scene(BaseModel):
    """One extracted drinking scene/context."""

    label: str = Field(..., description="中文场景短语，如「在家独饮」")
    en: str = Field(default="", description="英文小标，如「SOLO NIGHTS」")
    count: int = Field(default=0, description="该场景在这段聊天里出现的大致次数")
    vibe: str = Field(default="", description="一句话氛围/心情，如「放松」")


class MomentsResponse(BaseModel):
    """Structured drinking scenes summary."""

    scenes: List[Scene] = Field(default_factory=list)
