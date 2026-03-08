import os
from dotenv import load_dotenv
from langchain_core.messages import trim_messages
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage
from langgraph.checkpoint.memory import MemorySaver

from graph.state import AgentState
from tools.tools import AGENT_TOOLS

# load OPENAI_API_KEY from .env file
load_dotenv()

# initialize llm
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
llm_with_tools = llm.bind_tools(AGENT_TOOLS)


# agent node
def call_model(state: AgentState):
    """Passes the conversation history to the LLM"""
    system_prompt = SystemMessage("""
        You are a funny, brilliant assistant. Yuo have access to tools like web search and calculator. Use it if it makes sense!

        Rules:
        - If you are asked a factual question, ALWAYS use the web search tool to find the most up-to-date answer.
        - If you are asked to do math, ALWAYS use the calculator tool.
        - Be concise but friendly in your final answers.
    """)
    messages = state["messages"]

    # trim conversation history so context doesn't explode (sliding window)
    trimmed_messages = trim_messages(
        messages,
        max_tokens=2000,  # Adjust based on your model's limits
        strategy="last",  # Keep the most recent messages
        token_counter=llm,  # Use your exact OpenAI model to count tokens accurately
        allow_partial=False,  # Don't chop a single message in half
    )

    full_conversation = [system_prompt] + trimmed_messages

    response = llm_with_tools.invoke(full_conversation)
    # return dictionary that updates the "messages" state
    return {"messages": [response]}


# define the tool node
tool_node = ToolNode(
    AGENT_TOOLS
)  # automatically executes the functions the LLM asks for

# build the Graph
workflow = StateGraph(AgentState)

# add nodes to the graph
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

# set entry point (where the graph starts)
workflow.set_entry_point("agent")

# add edges
workflow.add_conditional_edges(
    "agent", tools_condition
)  # automatically check if the llm decided to call a tool. If yes, it routes to "tools". If no, it routes to END

# add edge between tools and agent.
# after the tool runs, route back to the agent so it can read the tool's output and formulate final answer
workflow.add_edge("tools", "agent")

memory = MemorySaver()

compiled_graph = workflow.compile(checkpointer=memory)
