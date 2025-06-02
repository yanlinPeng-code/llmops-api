#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/6/29 14:39
@Author  : thezehui@gmail.com
@File    : 2.TCVectorDB外部Embedding模式示例.py
"""
import os

import dotenv
from langchain_community.vectorstores import TencentVectorDB
from langchain_community.vectorstores.tencentvectordb import (
    ConnectionParams
)
from langchain_openai import OpenAIEmbeddings

dotenv.load_dotenv()

embedding = OpenAIEmbeddings(model="text-embedding-3-small")
db = TencentVectorDB(
    embedding=embedding,
    connection_params=ConnectionParams(
        url=os.environ.get("TC_VECTOR_DB_URL"),
        username=os.environ.get("TC_VECTOR_DB_USERNAME"),
        key=os.environ.get("TC_VECTOR_DB_KEY"),
        timeout=int(os.environ.get("TC_VECTOR_DB_TIMEOUT")),
    ),
    database_name=os.environ.get("TC_VECTOR_DB_DATABASE"),
    collection_name="dataset-external",
    # meta_fields=[
    #     MetaField(name="text", data_type=META_FIELD_TYPE_STRING),
    # ]
)

texts = [
    "笨笨是一只很喜欢睡觉的猫咪",
    "我喜欢在夜晚听音乐，这让我感到放松。",
    "猫咪在窗台上打盹，看起来非常可爱。",
    "学习新技能是每个人都应该追求的目标。",
    "我最喜欢的食物是意大利面，尤其是番茄酱的那种。",
    "昨晚我做了一个奇怪的梦，梦见自己在太空飞行。",
    "我的手机突然关机了，让我有些焦虑。",
    "阅读是我每天都会做的事情，我觉得很充实。",
    "他们一起计划了一次周末的野餐，希望天气能好。",
    "我的狗喜欢追逐球，看起来非常开心。",
]
# metadatas = [{"text": text, "page": index} for index, text in enumerate(texts)]

ids = db.add_texts(texts)
print("添加文档id列表:", ids)

print(db.similarity_search_with_score("我养了一只猫，叫笨笨"))
