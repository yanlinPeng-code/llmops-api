"""
1.从数据库中获取可用表
2.确定哪些表与问题相关
3.获取相关表的模式（schema）
4.根据问题和模式中的信息生成查询
5.使用 LLM 检查查询是否有常见错误
6.执行查询并返回结果
7.纠正数据库引擎指出的错误，直到查询成功
8.根据结果组织响应


"""
from dotenv import load_dotenv

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase

llm = init_chat_model("openai:gpt-4o")
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

for tool in tools:
    print(f"{tool.name}: {tool.description}\n")

# for tool in tools:
#     print(tool.name, tool.description)
from langgraph.prebuilt import create_react_agent

system_prompt = """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user
specifies a specific number of examples they wish to obtain, always limit your
query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

You MUST double check your query before executing it. If you get an error while
executing a query, rewrite the query and try again.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
database.

To start you should ALWAYS look at the tables in the database to see what you
can query. Do NOT skip this step.

Then you should query the schema of the most relevant tables.
""".format(
    dialect=db.dialect,
    top_k=5,
)

agent = create_react_agent(
    llm,
    tools,
    prompt=system_prompt,
)
question = "Which sales agent made the most in sales in 2009?"

for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
):
    print(step)
