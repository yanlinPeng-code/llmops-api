from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_openai import ChatOpenAI

load_dotenv()


def retrieval(query: str) -> str:
    """模拟一个检索器，传入查询query，输出文本"""
    print("执行检索:", query)
    return "我叫慕小课，是一名AI应用开发工程师"


# 1.编排Prompt
prompt = ChatPromptTemplate.from_template("""请根据用户的提问回答问题，可以参考对应的上下
文进行回复。
<context>
{context} <context> 用户的问题是: {query}""")
# 2.构建大语言模型
llm = ChatOpenAI(model="gpt-3.5-turbo")
# 3.创建输出解析器
parser = StrOutputParser()
# 4.编排链，RunnablePassthrough.assign写法
# chain = (
#         RunnablePassthrough.assign(context=lambda x: retrieval(x)) |
#         prompt |
#         llm |
#         parser
# )
chain = RunnableParallel(
    context=retrieval,
    query=RunnablePassthrough()

) | prompt | llm | parser

# 5.调用链生成结果
content = chain.invoke("你好，我叫什么?")
print(content)
