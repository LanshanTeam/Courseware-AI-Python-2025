#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : agent_with_tools.py
@Author  : Kevin
@Date    : 2025/10/29
@Description : Brief description of the file's purpose or functionality.
@Version : 1.0
"""

from agno.agent import Agent
from agno.tools.baidusearch import BaiduSearchTools

agent = Agent(
    tools=[BaiduSearchTools()],
    description="You are a search agent that helps users find the most relevant information using Baidu.",
    instructions=[
        "根据用户提供的主题，提供该主题最相关的三条搜索结果。",
        "搜索5个结果并选择排名前3的选项。",
        "回复字数限定在300以内。"
    ],
)
agent.print_response("介绍一下聚客AI学院?", markdown=True)
