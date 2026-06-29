"""Database enums."""

from enum import Enum


class PromptTypeEnum(str, Enum):
    """Prompt types matching Java PromptTypeEnum."""

    CHAT = "CHAT"
    IMAGE_RECOGNITION = "IMAGE_RECOGNITION"
    BARTENDER = "BARTENDER"
    TRANSLATE = "TRANSLATE"
    MATERIAL_ANALYSIS = "MATERIAL_ANALYSIS"
    CHAT_STREAM = "CHAT_STREAM"
    # 开喝时刻场景总结。和 CHAT_STREAM 一样只有 Python 读（Java 只转发），故 Python 端独有。
    MOMENTS = "MOMENTS"
