"""DrinkUp Backend integration tools for cocktail generation."""

import json
import logging
from typing import Type, Optional
import httpx
from pydantic import BaseModel, Field
from langchain.tools import BaseTool

from ..config import settings

logger = logging.getLogger(__name__)

# 免费额度用尽（后端返回 402）时，generate_cocktail 工具结果的前缀标志。
# 加载对话历史时用它识别并过滤掉「超额被拦」那一轮，避免模型学到用户超额、后续不再尝试出卡。
QUOTA_EXCEEDED_MARKER = "[QUOTA_EXCEEDED]"


class GenerateCocktailInput(BaseModel):
    """Input for generating a cocktail recipe. If using user's stock materials, include the details in the user_demand."""

    user_demand: str = Field(
        description="The user's cocktail requirements & flavor preferences, and user's stock materials details"
    )


class SearchCocktailInput(BaseModel):
    """Input for searching cocktail menu."""

    user_input: str = Field(
        description="The user's search query for cocktails (鸡尾酒搜索查询)"
    )


class GenerateCocktailTool(BaseTool):
    """Tool for generating cocktail recipes using DrinkUp backend."""

    name: str = "generate_cocktail"
    description: str = (
        "Generate a custom cocktail recipe based on user preferences. "
        "Use this when the user asks for cocktail recommendations, wants to create a new drink, "
        "or needs a recipe based on specific flavors or ingredients. "
        "Input should be the user's requirements or preferences in natural language."
    )
    args_schema: Type[BaseModel] = GenerateCocktailInput
    user_id: Optional[int] = None  # Store user_id for the API request

    def _run(self, user_demand: str) -> str:
        """Execute the tool synchronously."""
        import asyncio

        return asyncio.run(self._arun(user_demand))

    async def _arun(self, user_demand: str) -> str:
        """Generate a cocktail recipe based on user demand."""
        try:
            # Prepare the request according to OpenAPI spec
            url = f"{settings.drinkup_backend_url}/api/workflow/v2/bartender/public"
            payload = {
                "userDemand": user_demand,
                "user_id": self.user_id
                if self.user_id
                else 0,  # Use provided user_id or default to 0
            }

            logger.info(f"Calling DrinkUp backend API: {url}")
            logger.info(f"Request payload: {payload}")

            # Make the API call
            async with httpx.AsyncClient(
                timeout=settings.drinkup_backend_timeout
            ) as client:
                response = await client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )

                # Check response status
                if response.status_code != 200:
                    logger.error(
                        f"API call failed with status {response.status_code}: {response.text}"
                    )
                    return f"Failed to generate cocktail recipe. Status: {response.status_code}"

                # Parse response
                result = response.json()

                # Check business status code
                code = result.get("code")
                if code != 0:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Business error: {error_msg}")
                    # 402 = 今日免费额度用尽：带标志返回，供加载历史时把这一轮过滤掉，
                    # 避免模型「学到用户超额」后续不再尝试出卡（否则付费墙不再触发，影响转化）。
                    if code == 402:
                        return f"{QUOTA_EXCEEDED_MARKER} {error_msg}"
                    return f"Error generating cocktail: {error_msg}"

                # Extract cocktail data
                cocktail_data = result.get("data", {})

                # Return the raw JSON data as a string (whether it's array or object)
                if isinstance(cocktail_data, list) and len(cocktail_data) > 0:
                    logger.info(
                        f"Successfully generated {len(cocktail_data)} cocktail(s)"
                    )
                elif isinstance(cocktail_data, dict) and cocktail_data:
                    logger.info(
                        f"Successfully generated cocktail: {cocktail_data.get('name', 'Unknown')}"
                    )
                else:
                    logger.info("No cocktail data received")

                return json.dumps(cocktail_data, ensure_ascii=False)

        except httpx.TimeoutException:
            logger.error("Request timeout when calling DrinkUp backend")
            return "Request timeout. The backend service might be slow or unavailable."
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            return f"Failed to connect to backend service: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error generating cocktail: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"


class SearchCocktailTool(BaseTool):
    """Tool for searching cocktails from the menu."""

    name: str = "search_cocktail"
    description: str = (
        "Search for cocktails from the menu to get a list of cocktail options. "
        "Use this when the user wants to search cocktail menu, browse available cocktails, "
        "Input should be the user's search query in natural language."
    )
    args_schema: Type[BaseModel] = SearchCocktailInput

    def _run(self, user_input: str) -> str:
        """Execute the tool synchronously."""
        import asyncio

        return asyncio.run(self._arun(user_input))

    async def _arun(self, user_input: str) -> str:
        """Search cocktails from the menu."""
        try:
            # Prepare the request according to OpenAPI spec
            url = f"{settings.drinkup_backend_url}/api/workflow/cocktail"
            payload = {
                "userInput": user_input,
            }

            logger.info(f"Calling DrinkUp cocktail workflow API: {url}")
            logger.info(f"Request payload: {payload}")

            # Make the API call
            async with httpx.AsyncClient(
                timeout=settings.drinkup_backend_timeout
            ) as client:
                response = await client.post(
                    url, json=payload, headers={"Content-Type": "application/json"}
                )

                # Check response status
                if response.status_code != 200:
                    logger.error(
                        f"API call failed with status {response.status_code}: {response.text}"
                    )
                    return f"Failed to process cocktail request. Status: {response.status_code}"

                # Parse response
                result = response.json()

                # Check business status code
                if result.get("code") != 0:
                    error_msg = result.get("message", "Unknown error")
                    logger.error(f"Business error: {error_msg}")
                    return f"Error processing cocktail request: {error_msg}"

                # Return the raw response data as JSON string
                logger.info("Successfully searched cocktails from menu")
                return json.dumps(result, ensure_ascii=False)

        except httpx.TimeoutException:
            logger.error("Request timeout when searching cocktails")
            return "Request timeout. The backend service might be slow or unavailable."
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            return f"Failed to connect to backend service: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error searching cocktails: {str(e)}")
            return f"An unexpected error occurred: {str(e)}"


def create_drinkup_backend_tools(user_id: Optional[str] = None) -> list:
    """Create and return DrinkUp backend tools.

    Args:
        user_id: Optional user ID to associate with generated cocktails
    """
    # Convert string user_id to int if provided
    user_id_int = None
    if user_id:
        try:
            user_id_int = int(user_id)
        except (ValueError, TypeError):
            logger.warning(f"Invalid user_id format: {user_id}, using default")
            user_id_int = None

    tools = [
        GenerateCocktailTool(user_id=user_id_int),
        SearchCocktailTool(),
    ]

    logger.info(f"Created {len(tools)} DrinkUp backend tools for user_id: {user_id}")
    return tools
