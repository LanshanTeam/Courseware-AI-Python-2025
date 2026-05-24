#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : classifier.py
@Author  : Kevin
@Date    : 2025/10/26
@Description : 场景分类器.
@Version : 1.0
"""

import os
from openai import OpenAI
from config import SCENES

client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)


def classify_scene_by_keywords(query: str) -> str | None:
    """基于关键词快速匹配"""
    for scene, info in SCENES.items():
        if any(kw in query for kw in info["keywords"]):
            return scene
    return None


def classify_scene_by_llm(query: str) -> str:
    """LLM零样本分类（兜底）"""
    scene_names = list(SCENES.keys())
    scene_descs = [f"{k}: {v['name']}" for k, v in SCENES.items()]

    prompt = f"""
你是一个企业知识库路由系统。请根据用户问题，判断其最可能属于以下哪个业务场景：
{chr(10).join(scene_descs)}

问题：{query}
要求：只输出场景英文标识（如 hr, it, finance），不要解释。
"""
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        pred = response.choices[0].message.content.strip().lower()
        return pred if pred in SCENES else "hr"  # 默认回退到hr
    except Exception as e:
        print(f"LLM分类失败: {e}")
        return "hr"


def classify_scene(query: str) -> str:
    """主分类函数：先关键词，再LLM"""
    scene = classify_scene_by_keywords(query)
    if scene:
        return scene
    return classify_scene_by_llm(query)
