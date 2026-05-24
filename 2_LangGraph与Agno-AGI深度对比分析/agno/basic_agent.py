#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : basic_agent.py
@Author  : Kevin
@Date    : 2025/10/29
@Description : Basic Agent.
@Version : 1.0
"""

from agno.agent import Agent
from agno.models.openai import OpenAIChat

# 创建 Agent
agent = Agent(
    model=OpenAIChat(id="gpt-4o"),
    instructions="你是一个热情的新闻记者",
    markdown=True
)

agent.print_response("分享一则纽约新闻，只输出新闻的摘要信息，字数控制在100个以内")
