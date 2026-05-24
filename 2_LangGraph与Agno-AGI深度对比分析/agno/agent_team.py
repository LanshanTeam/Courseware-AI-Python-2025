#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : agent_team.py
@Author  : Kevin
@Date    : 2025/10/29
@Description : Brief description of the file's purpose or functionality.
@Version : 1.0
"""

# 从项目根目录或当前目录的 .env 加载 OPENAI_API_KEY（可选）
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from agno.team.team import Team

english_agent = Agent(
    name="English Agent",
    role="You only answer in English",
    model=OpenAIChat(id="gpt-4o"),
)
chinese_agent = Agent(
    name="Chinese Agent",
    role="You only answer in Chinese",
    model=DeepSeek(id="deepseek-chat"),
)


multi_language_team = Team(
    name="Multi Language Team",
    model=OpenAIChat("gpt-4o"),
    members=[english_agent, chinese_agent],
    markdown=True,
    description="You are a language router that directs questions to the appropriate language agent.",
    instructions=[
        "Identify the language of the user's question and direct it to the appropriate language agent.",
        "Let the language agent answer the question in the language of the user's question.",
        "If the user asks in a language whose agent is not a team member, respond in English with:",
        "'I can only answer in the following languages: English, Chinese. Please ask your question in one of these languages.'",
        "Always check the language of the user's input before routing to an agent.",
        "For unsupported languages like Italian, respond in English with the above message.",
    ],
    respond_directly=True,
    determine_input_for_members=False,
    show_members_responses=True,
)


if __name__ == "__main__":
    # Ask "How are you?" in all supported languages
    multi_language_team.print_response("How are you?", stream=True)  # English
    multi_language_team.print_response("你好吗？", stream=True)  # Chinese
    multi_language_team.print_response("Come stai?", stream=True)  # Italian
