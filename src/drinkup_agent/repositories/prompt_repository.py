"""Repository for PromptContent operations."""

from typing import Optional
from sqlalchemy import select

from ..database import get_async_session, PromptContent
from ..database.enums import PromptTypeEnum


class PromptRepository:
    """Repository for managing prompt content in database."""

    def __init__(self):
        self.session_factory = get_async_session()

    async def find_by_type(self, prompt_type: str) -> Optional[PromptContent]:
        """
        Find prompt content by type.

        Args:
            prompt_type: The type of prompt (e.g., 'CHAT', 'BARTENDER')

        Returns:
            PromptContent object if found, None otherwise
        """
        async with self.session_factory() as session:
            try:
                stmt = select(PromptContent).where(PromptContent.type == prompt_type)
                result = await session.execute(stmt)
                prompt = result.scalar_one_or_none()
                return prompt
            except Exception as e:
                print(f"Error fetching prompt by type {prompt_type}: {e}")
                return None

    async def get_chat_prompt(self) -> Optional[str]:
        """
        Get the chat system prompt from database.

        Returns:
            System prompt string if found, None otherwise
        """
        prompt = await self.find_by_type(PromptTypeEnum.CHAT_STREAM.value)
        return prompt.system_prompt if prompt else None

    async def get_moments_prompt(self) -> Optional[str]:
        """
        Get the 开喝时刻 summarization prompt from database.

        Returns:
            System prompt string if found, None otherwise (caller falls back to hardcoded).
        """
        prompt = await self.find_by_type(PromptTypeEnum.MOMENTS.value)
        return prompt.system_prompt if prompt else None
