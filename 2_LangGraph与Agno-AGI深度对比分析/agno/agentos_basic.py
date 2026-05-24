#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : agentos_basic.py
@Author  : Kevin
@Date    : 2025/10/29
@Description : AgentOS.
@Version : 1.0
"""

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.models.openai import OpenAIChat
from agno.team.team import Team
from agno.os import AgentOS

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

agent_os = AgentOS(
    id="my-first-os",
    description="My first AgentOS",
    teams=[multi_language_team],
)
app = agent_os.get_app()


if __name__ == "__main__":
    """Run your AgentOS.

    You can see the configuration and available apps at:
    http://localhost:7777/config

    """
    agent_os.serve(app="agentos_basic:app", reload=True)
