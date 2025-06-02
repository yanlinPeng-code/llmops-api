import os

from dotenv import load_dotenv

load_dotenv()
from langchain_community.utilities import SQLDatabase

from langchain_openai import ChatOpenAI

# llm = init_chat_model(model="deepseek:deepseek-chat")
llm = ChatOpenAI(model="qwq-plus", streaming=True, api_key=os.getenv("DASHSCOPE_API_KEY"),
                 base_url=os.getenv("DASHSCOPE_BASE_URL"))
db = SQLDatabase.from_uri("sqlite:///Chinook.db")
from langchain_community.agent_toolkits import SQLDatabaseToolkit

toolkit = SQLDatabaseToolkit(db=db, llm=llm)

tools = toolkit.get_tools()

from typing import Literal
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
get_schema_node = ToolNode([get_schema_tool], name="get_schema")

run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")
run_query_node = ToolNode([run_query_tool], name="run_query")


# Example: create a predetermined tool call
def list_tables(state: MessagesState):
    tool_call = {
        "name": "sql_db_list_tables",
        "args": {},
        "id": "abc123",
        "type": "tool_call",
    }
    tool_call_message = AIMessage(content="", tool_calls=[tool_call])

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    tool_message = list_tables_tool.invoke(tool_call)
    response = AIMessage(f"Available tables: {tool_message.content}")

    return {"messages": [tool_call_message, tool_message, response]}


# Example: force a model to create a tool call
def call_get_schema(state: MessagesState):
    # Note that LangChain enforces that all models accept `tool_choice="any"`
    # as well as `tool_choice=<string name of tool>`.
    llm_with_tools = llm.bind_tools([get_schema_tool], tool_choice="any")
    response = llm_with_tools.invoke(state["messages"])

    return {"messages": [response]}


generate_query_system_prompt = """
你是一个与 SQL 数据库交互的代理。
给定一个输入问题，创建一个语法正确的 {dialect} 查询并运行，
然后查看查询结果并返回答案。除非用户
指定了他们希望获取的特定示例数量，否则请始终将查询限制为最多 {top_k} 个结果。

你可以按相关列对结果进行排序，以返回数据库中最有趣的示例。切勿查询特定表中的所有列，
而应仅查询与问题相关的列。

请勿对数据库执行任何 DML 语句（INSERT、UPDATE、DELETE、DROP 等）。
""".format(
    dialect=db.dialect,
    top_k=5,
)


def generate_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": generate_query_system_prompt,
    }
    # We do not force a tool call here, to allow the model to
    # respond naturally when it obtains the solution.
    llm_with_tools = llm.bind_tools([run_query_tool])
    response = llm_with_tools.invoke([system_message] + state["messages"])

    return {"messages": [response]}


check_query_system_prompt = """
您是一位注重细节的 SQL 专家。
请仔细检查 {dialect} 查询中是否存在常见错误，包括：
- 将 NOT IN 与 NULL 值一起使用
- 应使用 UNION ALL 时却使用了 UNION
- 将 BETWEEN 用于排他范围
- 谓词中的数据类型不匹配
- 正确引用标识符
- 函数的参数数量正确
- 强制转换为正确的数据类型
- 连接时使用了正确的列

如果存在上述任何错误，请重写查询。如果没有错误，
只需重现原始查询即可。

运行此检查后，您将调用相应的工具来执行查询。
""".format(dialect=db.dialect)


def check_query(state: MessagesState):
    system_message = {
        "role": "system",
        "content": check_query_system_prompt,
    }

    # Generate an artificial user message to check
    tool_call = state["messages"][-1].tool_calls[0]
    user_message = {"role": "user", "content": tool_call["args"]["query"]}
    llm_with_tools = llm.bind_tools([run_query_tool], tool_choice="auto")
    response = llm_with_tools.invoke([system_message, user_message])
    response.id = state["messages"][-1].id

    return {"messages": [response]}


def should_continue(state: MessagesState) -> Literal[END, "check_query"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return END
    else:
        return "check_query"


builder = StateGraph(MessagesState)
builder.add_node(list_tables)
builder.add_node(call_get_schema)
builder.add_node(get_schema_node, "get_schema")
builder.add_node(generate_query)
builder.add_node(check_query)
builder.add_node(run_query_node, "run_query")

builder.add_edge(START, "list_tables")
builder.add_edge("list_tables", "call_get_schema")
builder.add_edge("call_get_schema", "get_schema")
builder.add_edge("get_schema", "generate_query")
builder.add_conditional_edges(
    "generate_query",
    should_continue,
)
builder.add_edge("check_query", "run_query")
builder.add_edge("run_query", "generate_query")

agent = builder.compile()
question = "2009 年哪位销售代理的销售额最高？"

for step in agent.stream(
        {"messages": [{"role": "user", "content": question}]},
        stream_mode="values",
):
    step["messages"][-1].pretty_print()
