from dotenv import load_dotenv
from langchain import hub
from langchain.chat_models import init_chat_model
from langchain_community.tools import QuerySQLDatabaseTool
from langchain_community.utilities import SQLDatabase
from langgraph.constants import START
from langgraph.graph import StateGraph
from typing_extensions import TypedDict, Annotated

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
load_dotenv()


class State(TypedDict):
    question: str
    query: str
    result: str
    answer: str


class QueryOut(TypedDict):
    """Generated SQL query."""
    query: Annotated[str, ..., "Syntactically valid SQL query."]


llm = init_chat_model("gpt-4o-mini", model_provider="openai")
"""

给定一个输入问题，创建一个语法正确的 {dialect} 查询来运行，以帮助找到答案。除非用户在问题中指定了他们希望获得的特定数量的示例，否则请始终将查询限制为最多 {top_k} 个结果。您可以按相关列对结果进行排序，以返回数据库中最有趣的示例。

切勿查询特定表中的所有列，而应仅查询与问题相关的少数几个列。

请注意仅使用您可以在架构描述中看到的列名。注意不要查询不存在的列。另外，还要注意哪个列位于哪个表中。

仅使用以下表：
{table_info}
"""
query_prompt_template = hub.pull("langchain-ai/sql-query-system-prompt")


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "top_k": 10,
            "table_info": db.get_table_info(),
            "input": state["question"]

        }
    )
    structured_llm = llm.with_structured_output(QueryOut)
    result = structured_llm.invoke(prompt)
    return {"query": result["query"]}


def execute_query(state: State) -> State:
    """Execute SQL query."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    return {"result": execute_query_tool.invoke(state["query"])}


def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "Given the following user question, corresponding SQL query, "
        "and SQL result, answer the user question.\n\n"
        f"Question:{state['question']}\n"
        f"Sql Query：{state['query']}\n"
        f"Sql Result：{state['result']}\n"
        "Answer:"

    )
    response = llm.invoke(prompt)
    return {"answer": response.content}


query = write_query({"question": "How many Employees are there?"})

graph_builder = StateGraph(State).add_sequence([write_query, execute_query, generate_answer])

graph_builder.add_edge(START, "write_query")
graph = graph_builder.compile()

for step in graph.stream({"question": "How many Employees are there?"}, stream_mode="updates"):
    print(step)
