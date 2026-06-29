"""User insights API: summarize drinking scenes (开喝时刻) from chat history."""

import logging

from fastapi import APIRouter

from ..models.insights import MomentsRequest, MomentsResponse
from ..services.insights_service import InsightsService

logger = logging.getLogger(__name__)
router = APIRouter()

# 无状态、轻量，进程内复用一个实例即可。
_service = InsightsService()


@router.post("/insights/moments", response_model=MomentsResponse)
async def summarize_moments(request: MomentsRequest) -> MomentsResponse:
    """一次性把用户近期聊天总结成 3–4 个开喝场景，返回结构化 JSON。"""
    logger.info(
        "[insights] moments request user_id=%s history=%d period=%s",
        request.user_id,
        len(request.history or []),
        request.period_label,
    )
    return await _service.summarize_moments(request.history or [])
