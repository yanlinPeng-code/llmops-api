from typing import TypedDict, Annotated, Any

from dotenv import load_dotenv
from langchain_community.chat_models import ChatTongyi
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

load_dotenv()


class State(TypedDict):
    messages: Annotated[list, add_messages]
    name: str


llm = ChatTongyi(model="qwen-max")


def chat_box(state: State, config: str) -> Any:
    ai_message = llm.invoke(state["messages"])
    return {"messages": [ai_message]}


graph_builder = StateGraph(State)
graph_builder.add_node("llm", chat_box)
graph_builder.set_entry_point("llm")
graph_builder.set_finish_point("llm")
# graph_builder.add_edge(START, "llm")
# graph_builder.add_edge("llm", END)

graph = graph_builder.compile()

res = graph.invoke({"messages": [("human", "你好，你是谁，我叫慕小课，我喜欢打篮球游泳")], "name": "小鹏"})
print(res)
