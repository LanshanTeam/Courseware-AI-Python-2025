#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@File    : config.py
@Author  : Kevin
@Date    : 2025/10/26
@Description : 定义场景.
@Version : 1.0
"""

SCENES = {
    "hr": {
        "name": "人力资源政策",
        "keywords": ["请假", "年假", "入职", "离职", "合同", "社保"],
        "path": "./data/hr"
    },
    "it": {
        "name": "IT支持",
        "keywords": ["电脑", "网络", "账号", "打印机", "软件", "VPN"],
        "path": "./data/it"
    },
    "finance": {
        "name": "财务报销",
        "keywords": ["报销", "发票", "差旅", "付款", "预算", "费用"],
        "path": "./data/finance"
    }
}
