"""Insights service: summarize a user's drinking scenes (开喝时刻) from chat history.

一次性、无状态的总结调用（不进对话历史、不写 Redis）：
把客户端带来的近期聊天喂给 LLM，让它提炼 3–4 个「开喝场景」，返回结构化 JSON。
刻意做得便宜、稳：低温度、限制条数、JSON 解析带兜底。
"""

import json
import logging
import re
from typing import List, Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from ..config import settings
from ..models.chat import ChatMessage
from ..models.insights import MomentsResponse, Scene
from ..repositories import PromptRepository

logger = logging.getLogger(__name__)

# 至少要有这么多条用户消息才值得总结，否则直接返回空（前端走空态）。
MIN_USER_TURNS = 3
# 喂给模型的最多条数（省 token；客户端通常已压缩过）。
MAX_HISTORY = 40
# 最多返回的场景数（按真实多样性给，不强行凑数）。
MAX_SCENES = 8

SYSTEM_PROMPT = """你是「开喝」App 的调酒师，正在帮用户回顾他最近这段时间的「开喝时刻」。
下面是用户和你（调酒师）的聊天记录。请你**只依据用户真实说过的话**，提炼出他最近出现过的「开喝场景」。

要求：
1. 把聊天里**真实出现过的场景都提炼出来**，按出现频率从高到低排序：真实有几个就给几个（最多 8 个）。用户聊天通常是多样的——别硬压成固定的 3-4 条，也别为了凑数编造没提过的场景。相近的场景合并成一个。
2. 每个场景给一个**中文短语 label**（4–12 字）。语气要**年轻、有网感、有态度**，像小红书标题/朋友圈文案那样会让人想笑或想转发，别写成正经流水账。
   - 多用情绪和梗：打工、摸鱼、emo、搞钱、续命、开摆、躺平、整顿、不上班、电子榨菜……怎么戳怎么来。
   - 反例（太无聊，别这样）：「下班解乏小酌一杯」「在家独自饮酒」「周末放松一下」。
   - 正例（要这种味儿）：「下班赎回我自己」「打工人的电子下酒菜」「emo 了就得喝」「搞钱间隙偷个闲」「熬夜续命水（含酒精）」「周五谁拦我」「假装在度假」「一个人的 happy hour」。
   - 仍要贴合用户真实说过的场景，只是把**措辞**写得更有意思，别为了梗编造没发生的事。
3. 每个场景给一个**英文小标 en**（全大写，2–3 个词，可以俏皮一点，例如「SOLO HAPPY HOUR」「AFTER WORK」「EMO O'CLOCK」「PAYDAY VIBES」）。
4. 给一个**大致次数 count**（整数，基于聊天里这类场景出现的频次，至少 1）。
5. 给一句**氛围 vibe**（2–4 字，例如「放松」「解压」「微醺」「热闹」「庆祝」「独处」）。
6. 不要编造用户没提过的场景；如果聊天内容很少或看不出场景，就只返回能确定的，宁缺毋滥。
7. 只输出 JSON，不要任何解释、不要 markdown 代码块。

输出格式（严格 JSON）：
{"scenes":[{"label":"下班赎回我自己","en":"AFTER WORK","count":6,"vibe":"解压"}]}"""


class InsightsService:
    """Stateless one-shot summarizer for drinking scenes."""

    def __init__(self, model: Optional[str] = None):
        # 复用聊天同款 OpenRouter 配置；可用 INSIGHTS_MODEL 单独指定更便宜的模型。
        chosen = model or settings.insights_model or settings.openai_model
        llm_params = {
            "model": chosen,
            "temperature": 0.4,
            "openai_api_key": settings.openai_api_key,
        }
        if settings.openai_base_url:
            llm_params["base_url"] = settings.openai_base_url
        self.llm = ChatOpenAI(**llm_params)
        # 优先用 prompt_content 表里管理的 MOMENTS prompt，读不到再用下面的硬编码兜底。
        self.prompt_repository = PromptRepository()

    def _format_transcript(self, history: List[ChatMessage]) -> str:
        lines = []
        for m in history[-MAX_HISTORY:]:
            who = "用户" if m.role == "user" else "调酒师"
            content = (m.content or "").strip()
            if content:
                lines.append(f"{who}：{content}")
        return "\n".join(lines)

    @staticmethod
    def _parse_scenes(raw: str) -> List[Scene]:
        """从模型输出里稳健地抠出 scenes（容忍 ```json 包裹 / 前后多余文字）。"""
        if not raw:
            return []
        text = raw.strip()
        # 去掉 markdown 代码块围栏
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        # 抠出第一个 { ... } JSON 对象
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return []
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            logger.warning("[insights] JSON 解析失败: %s", text[:200])
            return []
        scenes_raw = data.get("scenes") if isinstance(data, dict) else None
        if not isinstance(scenes_raw, list):
            return []
        scenes: List[Scene] = []
        for item in scenes_raw[:MAX_SCENES]:
            if not isinstance(item, dict) or not item.get("label"):
                continue
            try:
                count = int(item.get("count", 1))
            except (TypeError, ValueError):
                count = 1
            scenes.append(
                Scene(
                    label=str(item.get("label", "")).strip(),
                    en=str(item.get("en", "")).strip(),
                    count=max(1, count),
                    vibe=str(item.get("vibe", "")).strip(),
                )
            )
        return scenes

    async def summarize_moments(self, history: List[ChatMessage]) -> MomentsResponse:
        user_turns = sum(1 for m in history if m.role == "user" and (m.content or "").strip())
        if user_turns < MIN_USER_TURNS:
            logger.info("[insights] 用户消息过少(%d)，跳过总结", user_turns)
            return MomentsResponse(scenes=[])

        transcript = self._format_transcript(history)
        if not transcript:
            return MomentsResponse(scenes=[])

        # 表里有就用表里的（可在阿里云后台改），没有则用硬编码兜底
        db_prompt = await self.prompt_repository.get_moments_prompt()
        system_prompt = db_prompt or SYSTEM_PROMPT

        try:
            result = await self.llm.ainvoke(
                [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=f"聊天记录：\n{transcript}"),
                ]
            )
            content = result.content if isinstance(result.content, str) else str(result.content)
            scenes = self._parse_scenes(content)
            return MomentsResponse(scenes=scenes)
        except Exception as e:  # 模型/网络异常：返回空，让前端保留上次缓存
            logger.error("[insights] 总结失败: %s", e)
            return MomentsResponse(scenes=[])
