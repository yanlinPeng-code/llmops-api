#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Time    : 2024/7/4 8:06
@Author  : thezehui@gmail.com
@File    : 3.最大边际相关性示例.py
"""
import dotenv
import weaviate
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore
from weaviate.auth import AuthApiKey

dotenv.load_dotenv()

# 1.构建加载器与分割器
loader = UnstructuredMarkdownLoader("./项目API文档.md")
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "。|！|？", "\.\s|\!\s|\?\s", "；|;\s", "，|,\s", " ", "", ],
    is_separator_regex=True,
    chunk_size=500,
    chunk_overlap=50,
    add_start_index=True,
)

# 2.加载文档并分割
documents = loader.load()
chunks = text_splitter.split_documents(documents)

# 3.将数据存储到向量数据库
db = WeaviateVectorStore(
    client=weaviate.connect_to_wcs(
        cluster_url="https://eftofnujtxqcsa0sn272jw.c0.us-west3.gcp.weaviate.cloud",
        auth_credentials=AuthApiKey("21pzYy0orl2dxH9xCoZG1O2b0euDeKJNEbB0"),
    ),
    index_name="DatasetDemo",
    text_key="text",
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
)

# 4.执行最大边际相关性搜索
# search_documents = db.similarity_search("关于应用配置的接口有哪些？")
search_documents = db.max_marginal_relevance_search("关于应用配置的接口有哪些？")

# 5.打印搜索的结果
# print(list(document.page_content[:100] for document in search_documents))
for document in search_documents:
    print(document.page_content[:100])
    print("===========")
