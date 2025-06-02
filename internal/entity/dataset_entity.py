#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/8/25 14:34
@Author  : thezehui@gmail.com
@File    : dataset_entity.py
"""
from enum import Enum

# 默认知识库描述格式化文本
DEFAULT_DATASET_DESCRIPTION_FORMATTER = "当你需要回答管理《{name}》的时候可以引用该知识库。"


class ProcessType(str, Enum):
    """文档处理规则类型枚举"""
    AUTOMATIC = "automatic"
    CUSTOM = "custom"


# 默认的处理规则
DEFAULT_PROCESS_RULE = {
    "mode": "custom",
    "rule": {
        "pre_process_rules": [
            {"id": "remove_extra_space", "enabled": True},
            {"id": "remove_url_and_email", "enabled": True},
        ],
        "segment": {
            "separators": [
                "\n\n",
                "\n",
                "。|！|？",
                "\.\s|\!\s|\?\s",  # 英文标点符号后面通常需要加空格
                "；|;\s",
                "，|,\s",
                " ",
                ""
            ],
            "chunk_size": 500,
            "chunk_overlap": 50,
        }
    }
}


class DocumentStatus(str, Enum):
    """文档状态类型枚举"""
    WAITING = "waiting"
    PARSING = "parsing"
    SPLITTING = "splitting"
    INDEXING = "indexing"
    COMPLETED = "completed"
    ERROR = "error"


class SegmentStatus(str, Enum):
    """片段状态类型枚举"""
    WAITING = "waiting"
    INDEXING = "indexing"
    COMPLETED = "completed"
    ERROR = "error"


class RetrievalStrategy(str, Enum):
    """检索策略类型枚举"""
    FULL_TEXT = "full_text"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


class RetrievalSource(str, Enum):
    """检索来源"""
    HIT_TESTING = "hit_testing"
    APP = "app"
