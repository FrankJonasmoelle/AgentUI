import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

from graph.workflow import compiled_graph


st.set_page_config(page_title="LangGraph App", page_icon=None)
st.title("Your Personal Agent")

# initialize unique thread id for this session
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

# initialize chat history in streamlits session memory
if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hello, how can I help you today?")]

# display chat history on screen
for msg in st.session_state.messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# capture user input from chat box
if prompt := st.chat_input("Ask me anything or give me a task"):
    # append user message to memory and display it immediately
    user_msg = HumanMessage(content=prompt)
    st.session_state.messages.append(user_msg)
    with st.chat_message("user"):
        st.markdown(prompt)

    # pass conversation to langgraph and get the response
    with st.chat_message("assistant"):
        with st.spinner("Agent is thinking and using tools"):
            try:
                # Langgraph code
                config = {"configurable": {"thread_id": st.session_state.thread_id}}

                # stream response
                response_container = st.empty()
                full_response = ""

                stream = compiled_graph.stream(
                    {"messages": st.session_state.messages},
                    config,
                    stream_mode="messages",
                )

                for chunk, metadata in stream:
                    if (
                        metadata.get("langgraph_node") == "agent" and chunk.content
                    ):  # ignore chunks from the "tools node" or hidden tool calls
                        full_response += chunk.content
                        # update UI
                        response_container.markdown(full_response + "▌")

                response_container.markdown(
                    full_response
                )  # in the end, render without cursor ▌

                st.session_state.messages.append(AIMessage(content=full_response))

                # # run the graph, passing in the entire message history
                # new_state = compiled_graph.stream(
                #     {"messages": st.session_state.messages}, config
                # )

                # # extract final agent answer from the graph state
                # final_response = new_state["messages"][-1]

                # mock response
                # final_response = AIMessage(
                #     content=f"I received your message {prompt}. Implement agent code"
                # )

                # display response to user
                # st.markdown(final_response.content)

                # save agent response to chat history
                # st.session_state.messages.append(final_response)

            except Exception as e:
                st.error(f"Something went wrong: {e}")
