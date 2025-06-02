from dotenv import load_dotenv
from langchain_core.runnables import RunnablePassthrough

load_dotenv()
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase

db = SQLDatabase.from_uri("sqlite:///Chinook.db")
llm = init_chat_model("gpt-4o-mini", model_provider="openai")

from langchain_core.output_parsers.openai_tools import PydanticToolsParser
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


class Table(BaseModel):
    """Table in SQL database."""

    name: str = Field(description="Name of table in SQL database.")


table_names = "\n".join(db.get_usable_table_names())
system = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{table_names}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)
llm_with_tools = llm.bind_tools([Table])
output_parser = PydanticToolsParser(tools=[Table])

table_chain = prompt | llm_with_tools | output_parser

print(table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"}))
system = """Return the names of any SQL tables that are relevant to the user question.
The tables are:

Music
Business
"""

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system),
        ("human", "{input}"),
    ]
)

category_chain = prompt | llm_with_tools | output_parser

print(category_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"}))
from typing import List


def get_tables(categories: List[Table]) -> List[str]:
    tables = []
    for category in categories:
        if category.name == "Music":
            tables.extend(
                [
                    "Album",
                    "Artist",
                    "Genre",
                    "MediaType",
                    "Playlist",
                    "PlaylistTrack",
                    "Track",
                ]
            )
        elif category.name == "Business":
            tables.extend(["Customer", "Employee", "Invoice", "InvoiceLine"])
    return tables


table_chain = category_chain | get_tables

print(table_chain.invoke({"input": "What are all the genres of Alanis Morisette songs"}))
from operator import itemgetter
from langchain.chains.sql_database.query import create_sql_query_chain

query_chain = create_sql_query_chain(llm, db)
table_chain = {"input": itemgetter("question")} | table_chain

full_chain = RunnablePassthrough.assign(table_name_to_use=table_chain) | query_chain
query = full_chain.invoke({
    "question": "What are all the genres of Alanis Morisette songs"
})
print(query)
print(db.run(query))
