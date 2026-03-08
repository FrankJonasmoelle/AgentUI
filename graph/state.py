from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # tell langgraph to append new messages to the list instead of overwriting old ones
    messages: Annotated[list[AnyMessage], add_messages]
