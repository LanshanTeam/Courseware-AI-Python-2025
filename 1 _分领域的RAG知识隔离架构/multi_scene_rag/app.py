#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : app.py
@Author  : Kevin
@Date    : 2025/10/26
@Description : 主程序入口.
@Version : 1.0
"""

from rag_engine import MultiSceneRAG

if __name__ == "__main__":
    rag = MultiSceneRAG()

    while True:
        query = input("\n请输入您的问题（输入 'quit' 退出）: ")
        if query.lower() == 'quit':
            break
        answer = rag.query(query)
        print(f"✅ 答案:\n{answer}\n")