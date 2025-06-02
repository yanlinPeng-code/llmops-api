from typing import TypedDict

from dotenv import load_dotenv
from langgraph.graph import StateGraph, START

load_dotenv()


class State(TypedDict):
    topic: str
    joke: str


def refine_topic(state: State):
    return {"topic": state["topic"] + " and cats"}


from langchain_openai import ChatOpenAI

llm = ChatOpenAI(model="gpt-4o-mini")


def generate_joke(state: State):
    llm_response = llm.invoke(
        [
            {"role": "user", "content": f"Generate a joke about {state['topic']}"}
        ]
    )
    return {"joke": llm_response.content}


graph = (
    StateGraph(State)
    .add_node(refine_topic)
    .add_node(generate_joke)
    .add_edge(START, "refine_topic")
    .add_edge("refine_topic", "generate_joke")
    .compile()
)
for message_chunk, metadata in graph.stream(
        {"topic": "ice cream"},
        stream_mode="messages", ):
    print(message_chunk)
    # if message_chunk.content:
    #     print(message_chunk.content, end="|", flush=True)
