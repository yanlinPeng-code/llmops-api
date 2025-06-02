#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/18 9:04
@Author  : thezehui@gmail.com
@File    : 1.LangGraph实现CRAG.py
"""
from typing import Any

import dotenv
import weaviate
from langchain_community.chat_models import ChatTongyi
from langchain_community.tools import GoogleSerperRun
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain_weaviate import WeaviateVectorStore
from langgraph.graph import StateGraph
from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()


class GradeDocument(BaseModel):
    """文档评分模型"""
    score: str = Field(description="文档与问题是否关联，请回答yes或者no")


# class GradeDocument(BaseModel):
#     """文档评分Pydantic模型"""
#     binary_score: str = Field(description="文档与问题是否关联，请回答yes或者no")


class GoogleSerperArgsSchema(BaseModel):
    query: str = Field(description="执行谷歌搜索的查询语句")


class GraphState(TypedDict):
    """图结构应用程序状态"""
    question: str
    generation: str
    web_search: str
    documents: list[Document]


#
# class GraphState(TypedDict):
#     """图结构应用程序数据状态"""
#     question: str  # 原始问题
#     generation: str  # 大语言模型生成内容
#     web_search: str  # 网络搜索内容
#     documents: list[str]  # 文档列表


def format_docs(docs: list[Document]) -> str:
    """格式化传入的文档列表为字符串"""
    return "\n\n".join([doc.page_content for doc in docs])


# 1.创建大语言模型
llm = ChatTongyi(model="qwen-max")

# 2.创建检索器
vector_store = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="d5nfu9otssbnm7ocekjna.c0.asia-southeast1.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("DmAb7xWLJvWaZ1P1nILNvf7KWfZxFzNfVMcB"),
    ),
    index_name="DataSet",
    text_key="text",
    # embedding=DashScopeEmbeddings(model="text-embedding-v3"),
    embedding=OpenAIEmbeddings(model="text-embedding-3-small")
)
retriever = vector_store.as_retriever(search_type="mmr")
# 3.构建检索评估器
system = """你是一个检索评估到检索文档与问题相关性的评估员

如果检索到的文档与问题语义相关或者包含问题的关键字，将其评级为相关。

并给出一个是否相关的标志，如果相关给出得分为yes，不相关给出得分为no.

"""
grade_prompt = ChatPromptTemplate.from_messages(
    [("system", system),
     ("human", "检索到的文档：\n\n{document}\n\n用户的问题：{question}"),

     ]
)
retriever_grader = grade_prompt | llm.with_structured_output(GradeDocument)

# 4.RAg检索增强


template = """你是一个回答问题的助理，请根据上下文来回答问题，如果不知道，就只回答：抱歉，我不知道。并保持友好回答的风格

问题：{question}
上下文：{context}
答案：
"""
prompt = ChatPromptTemplate.from_template(
    template=template,
)
rag_chain = prompt | llm.bind(temperature=0) | StrOutputParser()

# 网络搜索问题进行重写

rewrite_prompt = ChatPromptTemplate.from_messages(
    [
        ("system",
         "你是一个将输入问题转化为更好的优化之后的问题的专业人士，将该输入问题优化后，方便进行该问题的网络检索，请查看输入问题，并分析推理其潜在语义意图/含义"),
        ("human", "这里是初始化问题：\n\n{question}\n\n请尝试提出一个改进的问题，要求该问题更加突出问题的中心含义")

    ]
)
question_rewriter = rewrite_prompt | llm.bind(temperature=0) | StrOutputParser()

# 网络搜素工具

google_serper = GoogleSerperRun(
    name="google_serper",
    description="一个低成本的谷歌搜索api,当你需要回答的问题关于时事更新的问题的时候，你可以调用该工具，查询相关信息，工具的输入是查询语句",
    args_schema=GoogleSerperArgsSchema,
    api_wrapper=GoogleSerperAPIWrapper(),
)


# 构建图相关的节点函数

def retrieve(state: GraphState) -> Any:
    """检索节点根据原始问题检索向量数据库"""

    print("***********检索节点**********")
    question = state["question"]
    documents = retriever.invoke(question)
    return {**state, "documents": documents}


# 生成节点

def generate(state: GraphState) -> Any:
    """llm生成节点，根据用户的原始问题+上下文内容调用llm生成内容"""
    print("*****llm生成内容*****")
    question = state["question"]
    documents = state["documents"]
    generation = rag_chain.invoke({"question": question, "context": format_docs(documents)})
    return {**state, "generation": generation}


def grade_documents(state: GraphState) -> Any:
    """文档与原始问题的关联性评分节点"""
    print("*****检查文档与问题关联性节点*****")
    question = state["question"]
    documents = state["documents"]
    web_search = "no"
    fitter_documents = []
    for document in documents:
        score: GradeDocument = retriever_grader.invoke({"question": question, "document": document.page_content})
        grade = score.score
        if grade.lower() == "yes":
            print("-------文档有关联-----")
            fitter_documents.append(document)
        else:
            print("-----文档没有关联-----")
            web_search = "yes"

    return {**state, "documents": fitter_documents, "web_search": web_search}


def transform_query(state: GraphState) -> Any:
    """转换问题重新检索节点"""
    print("*****重写查询节点***")
    question = state["question"]
    question = question_rewriter.invoke({"question": question})
    return {**state, "question": question}


def web_search(state: GraphState) -> Any:
    """web搜索节点"""
    print("*****进入web搜索节点")
    question = state["question"]
    documents = state["documents"]
    web_document = google_serper.invoke(question)
    documents.append(Document(
        page_content=web_document,
    ))
    return {**state, "documents": documents}


def choice_to_generation(state: GraphState) -> Any:
    """路由"""
    print("------路由选择边—----")
    web_search = state["web_search"]
    if web_search == "yes":
        print("------执行web搜索---")
        return "transform_query"

    else:
        print("*****直接生成****")
        return "generate"


# 构建图和边

workflow = StateGraph(GraphState)
# 添加节点
workflow.add_node("retrieve", retrieve)
workflow.add_node("grade_documents", grade_documents)
workflow.add_node("transform_query", transform_query)
workflow.add_node("web_search_node", web_search)
workflow.add_node("generate", generate)
# 添加条件边

workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "grade_documents")
workflow.add_conditional_edges("grade_documents", choice_to_generation)
workflow.add_edge("transform_query", "web_search_node")
workflow.add_edge("web_search_node", "generate")
workflow.set_finish_point("generate")

work = workflow.compile()

print(work.invoke({"question": "能介绍下什么是llmops吗"}))
