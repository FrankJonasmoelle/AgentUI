from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

web_search = DuckDuckGoSearchRun()


@tool
def multiply_numbers(a: float, b: float) -> float:
    """Used to multiply two numbers together. Use it for math"""
    return a * b


# group tools into list so they can be passed to agent later
AGENT_TOOLS = [multiply_numbers, web_search]
