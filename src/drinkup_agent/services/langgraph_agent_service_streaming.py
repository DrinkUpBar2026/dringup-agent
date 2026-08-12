"""Enhanced LangGraph agent service with character-level streaming."""

import json
import uuid
import logging
import re
from typing import Optional, List, Dict, Any, AsyncIterator

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain.tools import BaseTool

# Tool execution helper: Try to import ToolNode; if unavailable, we fallback later
try:
    from langgraph.prebuilt import ToolNode  # type: ignore
except Exception:  # ModuleNotFoundError or others
    ToolNode = None  # Fallback to manual tool execution

from ..config import settings
from ..repositories import PromptRepository
from ..services.conversation_service import ConversationService
from ..tools.mem0_tools import create_mem0_tools
from ..tools.drinkup_backend_tools import create_drinkup_backend_tools, QUOTA_EXCEEDED_MARKER

logger = logging.getLogger(__name__)


class StreamingLangGraphAgentService:
    """Service for managing LangGraph agents with character-level streaming."""

    def __init__(self):
        """Initialize the agent service."""
        self.prompt_repository = PromptRepository()
        self.conversation_service = ConversationService()
        self.model = settings.openai_model

        # Initialize streaming LLM with usage metadata enabled
        llm_params = {
            "model": self.model,
            "temperature": 0.7,
            "openai_api_key": settings.openai_api_key,
            "streaming": True,
            "stream_options": {
                "include_usage": True
            },  # Enable usage data in streaming response
        }

        if settings.openai_base_url:
            llm_params["base_url"] = settings.openai_base_url

        self.llm = ChatOpenAI(**llm_params)

    async def _get_tools(self, user_id: str, language: Optional[str] = None) -> List[BaseTool]:
        """Get all available tools for the agent."""
        tools = []

        # Add Mem0 tools only when explicitly enabled. Local core-flow debugging
        # should not be blocked by memory/embedding configuration.
        if settings.memory_enabled:
            mem0_tools = create_mem0_tools(
                api_key=settings.openai_api_key, user_id=user_id
            )
            tools.extend(mem0_tools)

        # Add DrinkUp backend tools with user_id and response language.
        drinkup_tools = create_drinkup_backend_tools(user_id=user_id, language=language)
        tools.extend(drinkup_tools)

        return tools

    async def _build_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        db_prompt = await self.prompt_repository.get_chat_prompt()

        default_prompt = """You are a knowledgeable and friendly bartender assistant for DrinkUp.
You help users with cocktail recommendations, recipes, and bar management.

You have access to these tools:

**Memory Management Tools:**
- Use 'add_memory' to store important information about the user
- Use 'search_memory' to recall relevant information when answering questions
- Use 'get_all_memories' to see everything you know about the user
- Use 'update_memory' to correct or update stored information
- Use 'delete_memory' to remove outdated information

**Cocktail Generation Tool:**
- Use 'generate_cocktail' when users ask for cocktail recommendations or custom recipes
- This tool creates detailed cocktail recipes based on user preferences, flavors, or ingredients

Always:
1. When users ask for cocktail recommendations, use the generate_cocktail tool
2. Search for relevant memories when starting a conversation or when context would be helpful
3. Store new important information shared by the user (preferences, restrictions, favorites, etc.)
4. Update memories when users correct or change their preferences
5. Provide personalized responses based on stored memories and generated cocktails

Current context:
- User Stock: {user_stock}
- User Info: {user_info}

Respond naturally without JSON formatting."""

        return db_prompt if db_prompt else default_prompt

    async def chat_with_agent_stream(
        self,
        user_message: str,
        user_id: str,
        conversation_id: Optional[str] = None,
        user_stock: Optional[str] = None,
        user_info: Optional[str] = None,
        image_attachments: Optional[List[Dict[str, Any]]] = None,
        history: Optional[List[Any]] = None,
        language: Optional[str] = None,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        Process a chat message with character-level streaming.
        """
        # Generate conversation_id if not provided (first message scenario)
        is_new_conversation = False
        if not conversation_id:
            conversation_id = str(uuid.uuid4())
            is_new_conversation = True
            logger.info(f"Generated new conversation_id: {conversation_id}")

        # Cold-start re-seed: the server-side conversation cache only lives 24h.
        # When it has expired (or this is a new id) but the client carried prior
        # turns, seed them so the agent regains context. Skip if Redis still has it.
        if history:
            try:
                existing = await self.conversation_service.get_messages(conversation_id)
                if not existing:
                    seeded = 0
                    for m in history:
                        role = getattr(m, "role", None) or (
                            m.get("role") if isinstance(m, dict) else None
                        )
                        content = getattr(m, "content", None) or (
                            m.get("content") if isinstance(m, dict) else None
                        )
                        if role and content:
                            await self.conversation_service.add_message(
                                conversation_id, role, content
                            )
                            seeded += 1
                    logger.info(
                        f"Re-seeded {seeded} message(s) into conversation {conversation_id} from client history"
                    )
            except Exception as seed_err:
                logger.warning(f"Failed to re-seed conversation history: {seed_err}")

        try:
            # Prepare a LangChain/LangSmith run config so all runs carry the same thread ID
            # This enables grouping runs into a single LangSmith Thread.
            run_config = {
                "metadata": {
                    "ls_thread_id": conversation_id,  # LangSmith-recognized thread id
                    "conversation_id": conversation_id,  # Also store for convenience/queries
                }
            }

            # If this is a new conversation, immediately save the user message
            if is_new_conversation:
                logger.info(f"Saving first user message to conversation {conversation_id}")
                await self.conversation_service.add_message(conversation_id, "user", user_message)

            # Prepare chat history (past conversation + current user message)
            past_messages = await self.conversation_service.get_messages(conversation_id)
            chat_history = self._convert_history_to_langchain(past_messages)

            # Only append current user message if it's not already in history
            user_msg = self._build_user_message(user_message, image_attachments)
            if not is_new_conversation:
                chat_history.append(user_msg)

            # Get tools and system prompt
            tools = await self._get_tools(user_id, language)
            system_prompt = await self._build_system_prompt()
            formatted_prompt = self._format_prompt(system_prompt, user_info, user_stock, language)

            # Create messages with system prompt
            messages_with_system = [SystemMessage(content=formatted_prompt)] + chat_history

            # Keep the locale constraint adjacent to the latest user turn. The base bartender
            # prompt and user context can be long and Chinese-heavy, so a single instruction at
            # the end of that prompt is not reliable enough for every model response.
            language_name = self.LANGUAGE_NAMES.get(language or "")
            if language_name:
                language_guard = SystemMessage(
                    content=(
                        f"MANDATORY RESPONSE LOCALE: {language_name}. "
                        f"Answer the latest user request only in {language_name}. "
                        "Do not copy the language of inventory, profile, memory, tool output, or "
                        "the base bartender prompt. "
                        f"{self.LANGUAGE_STYLE_RULES.get(language or '', '')}"
                    )
                )
                insert_at = len(messages_with_system)
                if chat_history and isinstance(chat_history[-1], HumanMessage):
                    insert_at -= 1
                messages_with_system.insert(insert_at, language_guard)

            # Bind tools to LLM
            llm_with_tools = self.llm.bind_tools(tools) if tools else self.llm

            # Stream the first response (may contain tool calls)
            full_content = ""
            is_tool_call = False
            quota_blocked_turn = False  # 本轮出卡是否因免费额度用尽被拦
            accumulated_message = None
            usage_data = None
            finish_reason = None

            async for chunk in llm_with_tools.astream(messages_with_system, config=run_config):
                logger.info(f"Chunk: {chunk}")
                if accumulated_message is None:
                    accumulated_message = chunk
                else:
                    accumulated_message += chunk

                if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                    finish_reason = chunk.response_metadata.get("finish_reason")

                if hasattr(chunk, "usage_metadata"):
                    usage_data = chunk.usage_metadata
                elif (
                    hasattr(chunk, "response_metadata") and "token_usage" in chunk.response_metadata
                ):
                    usage_data = chunk.response_metadata["token_usage"]

                if hasattr(chunk, "tool_calls") and chunk.tool_calls:
                    # Mark that tool calls are present, but do not emit yet.
                    # We will wait until the tool call arguments are fully accumulated.
                    is_tool_call = True
                elif hasattr(chunk, "content") and chunk.content:
                    content_chunk = chunk.content
                    full_content += content_chunk
                    if not language_name:
                        yield {"type": "streaming_content", "data": {"content": content_chunk}}

            if language_name and full_content:
                full_content = await self._ensure_response_language(
                    full_content, language or "", run_config
                )
                yield {"type": "streaming_content", "data": {"content": full_content}}

            if finish_reason in ["stop", "tool_calls"] and full_content:
                yield {"type": "final_message", "data": {"content": full_content}}

            # If the model decided to call tools, surface and execute them
            if is_tool_call and accumulated_message and hasattr(accumulated_message, "tool_calls"):
                # Save AI message with tool calls to conversation
                await self._save_ai_message_with_tool_calls(conversation_id, accumulated_message)

                # Emit agent_thinking after all tool call params are fully collected
                tool_calls_data = self._extract_tool_calls_for_thinking(accumulated_message)
                if tool_calls_data:
                    yield {"type": "agent_thinking", "data": {"tool_calls": tool_calls_data}}

                # Prepare to execute tools
                messages_with_system.append(accumulated_message)
                tool_results_messages, tool_events = await self._execute_tools(
                    tools, accumulated_message, messages_with_system, conversation_id
                )

                # Yield each tool result event
                for ev in tool_events:
                    yield ev

                # 检测本轮出卡是否因免费额度用尽被拦（工具结果带超额标志）
                quota_blocked_turn = any(
                    ev.get("type") == "tool_result"
                    and (ev.get("data") or {}).get("tool") == "generate_cocktail"
                    and QUOTA_EXCEEDED_MARKER in ((ev.get("data") or {}).get("result") or "")
                    for ev in tool_events
                )

                # Add tool results to messages and ask for final response
                messages_with_system.extend(tool_results_messages)

                final_content, final_usage_data, final_finish_reason = (
                    "",
                    None,
                    None,
                )
                async for chunk in self.llm.astream(messages_with_system, config=run_config):
                    if hasattr(chunk, "response_metadata") and chunk.response_metadata:
                        final_finish_reason = chunk.response_metadata.get("finish_reason")
                    if hasattr(chunk, "usage_metadata"):
                        final_usage_data = chunk.usage_metadata
                    elif (
                        hasattr(chunk, "response_metadata") and "token_usage" in chunk.response_metadata
                    ):
                        final_usage_data = chunk.response_metadata["token_usage"]
                    if hasattr(chunk, "content") and chunk.content:
                        content_chunk = chunk.content
                        final_content += content_chunk
                        if not language_name:
                            yield {"type": "streaming_content", "data": {"content": content_chunk}}

                if language_name and final_content:
                    final_content = await self._ensure_response_language(
                        final_content, language or "", run_config
                    )
                    yield {"type": "streaming_content", "data": {"content": final_content}}

                if final_finish_reason == "stop":
                    yield {"type": "final_message", "data": {"content": final_content}}

                usage_data = self._merge_usage(usage_data, final_usage_data)
                full_content = final_content or full_content

            # Persist conversation turns (user message already saved if new conversation)。
            # 超额被拦那一轮不落库 user/assistant，配合历史加载时过滤掉工具调用，
            # 让这一轮在记忆里「不留痕迹」——否则模型记住用户超额，之后不再尝试出卡，付费墙就不再弹（影响转化）。
            if not quota_blocked_turn:
                if not is_new_conversation:
                    await self.conversation_service.add_message(conversation_id, "user", user_message)
                await self.conversation_service.add_message(conversation_id, "assistant", full_content)

            # Compute usage for final response
            usage = (
                self._usage_from_provider(usage_data)
                if usage_data
                else self._estimate_usage(messages_with_system, full_content)
            )

            yield {
                "type": "final_response",
                "data": {
                    "conversation_id": conversation_id,
                    "content": full_content,
                    "usage": usage,
                },
            }

        except Exception as e:
            import traceback
            logger.error("Error in streaming agent chat: %s", str(e))
            logger.error("Full traceback: %s", traceback.format_exc())
            yield {
                "type": "error",
                "data": {
                    "conversation_id": conversation_id,
                    "content": f"I encountered an error processing your request: {str(e)}",
                    "usage": None,
                },
            }

    def _convert_history_to_langchain(self, messages: List[Dict[str, Any]]):
        """Convert stored conversation messages to LangChain message objects."""
        # 先找出「超额被拦」的工具结果（带标志），收集其 tool_call_id。
        # 这些 generate_cocktail 调用因免费额度用尽而失败，要从历史里成对剔除（工具调用 + 结果），
        # 否则模型会学到用户超额、之后不再尝试出卡，付费墙就不再触发。
        quota_blocked_ids = set()
        for msg in messages:
            if msg.get("role") == "tool" and QUOTA_EXCEEDED_MARKER in (msg.get("content") or ""):
                tcid = (msg.get("metadata") or {}).get("tool_call_id")
                if tcid:
                    quota_blocked_ids.add(tcid)

        chat_history: List[Any] = []
        for msg in messages:
            role = msg.get("role")
            if role == "user":
                chat_history.append(HumanMessage(content=msg.get("content", "")))
            elif role == "assistant":
                metadata = msg.get("metadata", {})
                tool_calls = metadata.get("tool_calls", [])
                # 剔除超额被拦的工具调用，保留模型当时说的话
                if tool_calls and quota_blocked_ids:
                    tool_calls = [tc for tc in tool_calls if tc.get("id") not in quota_blocked_ids]
                content = msg.get("content", "")
                if tool_calls:
                    chat_history.append(AIMessage(content=content, tool_calls=tool_calls))
                elif content:
                    chat_history.append(AIMessage(content=content))
                # 否则（工具调用被剔空且无文本的超额轮）：整条跳过，不留痕迹
            elif role == "tool":
                metadata = msg.get("metadata", {})
                tool_call_id = metadata.get("tool_call_id")
                # 跳过超额被拦的工具结果
                if tool_call_id and tool_call_id in quota_blocked_ids:
                    continue
                if tool_call_id:
                    chat_history.append(
                        ToolMessage(content=msg.get("content", ""), tool_call_id=tool_call_id)
                    )
        return chat_history

    def _build_user_message(
        self, user_message: str, image_attachments: Optional[List[Dict[str, Any]]]
    ) -> HumanMessage:
        """Build a HumanMessage, supporting optional image attachments."""
        if image_attachments:
            message_content: List[Dict[str, Any]] = [{"type": "text", "text": user_message}]
            for attachment in image_attachments:
                if hasattr(attachment, "image_base64"):
                    mime_type = getattr(attachment, "mime_type", None) or "image/jpeg"
                    image_base64 = getattr(attachment, "image_base64", None)
                else:
                    mime_type = attachment.get("mime_type", "image/jpeg")
                    image_base64 = attachment.get("image_base64")
                if not image_base64:
                    logger.warning("Skipping image attachment without base64 data")
                    continue
                image_data = {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                }
                message_content.append(image_data)
            return HumanMessage(content=message_content)
        return HumanMessage(content=user_message)

    LANGUAGE_NAMES = {"en": "English", "ja": "Japanese (日本語)", "ko": "Korean (한국어)"}
    LANGUAGE_STYLE_RULES = {
        "en": "Use English only. Do not include Chinese words or Chinese characters.",
        "ja": (
            "Use natural, idiomatic Japanese throughout. Japanese kanji is allowed where normal, "
            "but do not use Simplified-Chinese-only vocabulary, Chinese ingredient names, or "
            "Chinese-style literal translations of cocktail names. Write established cocktail "
            "names and foreign ingredient names in the standard Japanese katakana used in Japan "
            "(for example, write White Lady as ホワイト・レディ, never 白色淑女)."
        ),
        "ko": (
            "Use natural Korean throughout. Do not use Chinese characters (hanja) or Chinese-only "
            "words; transliterate ingredient names into Hangul."
        ),
    }
    LANGUAGE_FALLBACKS = {
        "en": "Sorry, I couldn't prepare an English response. Please try again.",
        "ja": "申し訳ありません。日本語の応答を生成できませんでした。もう一度お試しください。",
        "ko": "죄송합니다. 한국어 답변을 생성하지 못했습니다. 다시 시도해 주세요.",
    }

    @staticmethod
    def _has_language_mismatch(content: str, language: str) -> bool:
        han = re.search(r"[\u3400-\u4dbf\u4e00-\u9fff]", content)
        kana = re.search(r"[\u3040-\u30ff]", content)
        hangul = re.search(r"[\uac00-\ud7af]", content)
        if language == "en":
            return bool(han or kana or hangul)
        if language == "ko":
            return bool(han or kana)
        if language == "ja":
            return bool(hangul or (han and not kana))
        return False

    async def _ensure_response_language(
        self, content: str, language: str, run_config: Dict[str, Any]
    ) -> str:
        if not self._has_language_mismatch(content, language):
            return content

        language_name = self.LANGUAGE_NAMES.get(language, language)
        logger.warning("Detected mixed-language response for locale %s; rewriting", language)
        candidate = content
        for attempt in range(2):
            repair_messages = [
                SystemMessage(
                    content=(
                        f"Rewrite the supplied assistant response entirely in {language_name}. "
                        "Preserve its meaning and any <options> structure, but do not add facts. "
                        f"{self.LANGUAGE_STYLE_RULES.get(language, '')}"
                    )
                ),
                HumanMessage(content=candidate),
            ]
            repaired = await self.llm.ainvoke(repair_messages, config=run_config)
            repaired_content = getattr(repaired, "content", "")
            if isinstance(repaired_content, str) and repaired_content.strip():
                candidate = repaired_content.strip()
                if not self._has_language_mismatch(candidate, language):
                    logger.info(
                        "Mixed-language response repaired for locale %s on attempt %s",
                        language,
                        attempt + 1,
                    )
                    return candidate

        logger.error("Could not repair response language for locale %s", language)
        return self.LANGUAGE_FALLBACKS.get(language, content)

    def _format_prompt(
        self,
        system_prompt: str,
        user_info: Optional[str],
        user_stock: Optional[str],
        language: Optional[str] = None,
    ) -> str:
        """Interpolate user context placeholders in the system prompt."""
        formatted = system_prompt
        formatted = formatted.replace("{userInfo}", user_info or "Not provided")
        formatted = formatted.replace("{userStock}", user_stock or "Not provided")
        formatted = formatted.replace("{user_info}", user_info or "Not provided")
        formatted = formatted.replace("{user_stock}", user_stock or "Not provided")
        language_name = self.LANGUAGE_NAMES.get(language or "")
        if language_name:
            formatted += (
                f"\n\nIMPORTANT: The user's app language is {language_name}. "
                f"Always reply in {language_name}, regardless of the language used above. "
                f"Keep every sentence, suggestion label, and ingredient name in one consistent language. "
                f"{self.LANGUAGE_STYLE_RULES.get(language or '', '')}"
            )
        return formatted

    def _extract_tool_calls_for_thinking(self, accumulated_message: Any) -> List[Dict[str, Any]]:
        """Extract simplified tool call info for UI 'agent_thinking' event."""
        tool_calls_data: List[Dict[str, Any]] = []
        for tc in getattr(accumulated_message, "tool_calls", []) or []:
            tool_calls_data.append({"name": tc.get("name"), "args": tc.get("args")})
        return tool_calls_data

    async def _save_ai_message_with_tool_calls(self, conversation_id: str, accumulated_message: Any) -> None:
        """Persist AI message that initiated tool calls (with raw tool_calls)."""
        await self.conversation_service.add_ai_message_with_tool_calls(
            conversation_id,
            "",
            getattr(accumulated_message, "tool_calls", []),
        )

    async def _execute_tools(
        self,
        tools: List[BaseTool],
        accumulated_message: Any,
        messages_with_system: List[Any],
        conversation_id: str,
    ) -> (List[ToolMessage], List[Dict[str, Any]]):
        """Execute tool calls and return ToolMessages plus UI events to emit."""
        # Execute using ToolNode when available, otherwise manual fallback
        if ToolNode is not None:
            tool_node = ToolNode(tools)
            # Propagate LangSmith thread metadata into tool execution
            tool_run_config = {
                "configurable": {},
                "metadata": {
                    "ls_thread_id": conversation_id,
                    "conversation_id": conversation_id,
                },
            }
            tool_results = await tool_node.ainvoke({"messages": messages_with_system}, tool_run_config)
        else:
            tool_messages: List[ToolMessage] = []
            for tc in getattr(accumulated_message, "tool_calls", []) or []:
                tool_name = tc.get("name")
                tool_args = tc.get("args", {})
                tool_id = tc.get("id")
                selected = None
                for t in tools:
                    try:
                        if getattr(t, "name", None) == tool_name:
                            selected = t
                            break
                    except Exception:
                        pass
                if selected is None:
                    result_text = f"Tool '{tool_name}' not found"
                else:
                    try:
                        if hasattr(selected, "ainvoke"):
                            # Pass metadata so tool run is associated to the same thread
                            result = await selected.ainvoke(tool_args, config={
                                "metadata": {
                                    "ls_thread_id": conversation_id,
                                    "conversation_id": conversation_id,
                                }
                            })
                        elif hasattr(selected, "arun"):
                            result = await selected.arun(**tool_args)
                        else:
                            result = selected.run(**tool_args)
                        result_text = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
                    except Exception as e:
                        result_text = f"Error running tool '{tool_name}': {e}"
                tool_messages.append(ToolMessage(content=result_text, tool_call_id=tool_id))
            tool_results = {"messages": tool_messages}

        events: List[Dict[str, Any]] = []
        # Persist tool results and prepare UI events
        if "messages" in tool_results:
            for msg in tool_results["messages"]:
                if isinstance(msg, ToolMessage):
                    tool_name = None
                    for tc in getattr(accumulated_message, "tool_calls", []) or []:
                        if tc.get("id") == msg.tool_call_id:
                            tool_name = tc.get("name")
                            break
                    await self.conversation_service.add_tool_result(
                        conversation_id,
                        msg.tool_call_id,
                        tool_name or "unknown",
                        msg.content,
                    )
                    events.append(
                        {
                            "type": "tool_result",
                            "data": {"tool": tool_name or "unknown", "result": msg.content},
                        }
                    )
        return tool_results.get("messages", []), events

    def _merge_usage(self, first: Optional[Dict[str, Any]], second: Optional[Dict[str, Any]]):
        if first and second:
            return {
                "input_tokens": first.get("input_tokens", 0) + second.get("input_tokens", 0),
                "output_tokens": first.get("output_tokens", 0) + second.get("output_tokens", 0),
                "total_tokens": first.get("total_tokens", 0) + second.get("total_tokens", 0),
            }
        return second or first


    def _usage_from_provider(self, usage_data: Dict[str, Any]) -> Dict[str, int]:
        return {
            "prompt_tokens": usage_data.get("input_tokens", 0),
            "completion_tokens": usage_data.get("output_tokens", 0),
            "total_tokens": usage_data.get("total_tokens", 0),
        }

    def _estimate_usage(self, messages_with_system: List[Any], full_content: str) -> Dict[str, int]:
        prompt_tokens = sum(len(getattr(m, "content", "").split()) * 1.3 for m in messages_with_system)
        completion_tokens = len(full_content.split()) * 1.3
        return {
            "prompt_tokens": int(prompt_tokens),
            "completion_tokens": int(completion_tokens),
            "total_tokens": int(prompt_tokens + completion_tokens),
        }
