#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/6 11:09
@Author  : thezehui@gmail.com
@File    : 2.多向量索引-假设性查询检索原文档.py
"""
from typing import List

import dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

dotenv.load_dotenv()


class HypotheticalQuestions(BaseModel):
    """生成假设性问题"""
    questions: List[str] = Field(
        description="假设性问题列表，类型为字符串列表",
    )


# 1.构建一个生成假设性问题的prompt
prompt = ChatPromptTemplate.from_template("生成一个包含3个假设性问题的列表，这些问题可以用于回答下面的文档:\n\n{doc}")

# 2.创建大语言模型，并绑定对应的规范化输出结构
llm = ChatOpenAI(model="gpt-3.5-turbo-16k", temperature=0)
structured_llm = llm.with_structured_output(HypotheticalQuestions)

# 3.创建链应用
chain = (
        {"doc": lambda x: x.page_content}
        | prompt
        | structured_llm
)

hypothetical_questions: HypotheticalQuestions = chain.invoke(
    Document(page_content="我叫慕小课，我喜欢打篮球，游泳")
)
print(hypothetical_questions)
