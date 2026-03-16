# server.py
import json
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from graph.workflow import compiled_graph

app = FastAPI(title="LangGraph Agent API")

# allows Next.js frontend (running on port 3000)
# to talk to this FastAPI backend (running on port 8000).
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*"
    ],  # In production, change this to your exact frontend URL (e.g., ["http://localhost:3000"])
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# define the data we expect from the frontend
class ChatRequest(BaseModel):
    message: str
    thread_id: str


# create the chat endpoint
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):

    # We use an async generator to yield text chunks as they arrive
    async def stream_generator():
        user_msg = HumanMessage(content=request.message)
        config = {"configurable": {"thread_id": request.thread_id}}

        try:
            # Notice we use .astream() instead of .stream() for async FastAPI!
            async for chunk, metadata in compiled_graph.astream(
                {"messages": [user_msg]}, config, stream_mode="messages"
            ):
                # Filter for just the agent's conversational text
                if metadata.get("langgraph_node") == "agent" and chunk.content:
                    # Package the chunk into a JSON string and yield it in SSE format
                    data = json.dumps({"text": chunk.content})
                    yield f"data: {data}\n\n"

        except Exception as e:
            # Send an error message if something breaks
            error_data = json.dumps({"error": str(e)})
            yield f"data: {error_data}\n\n"

    # Return the generator wrapped in a StreamingResponse
    return StreamingResponse(stream_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    # Run the server on port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8080, reload=True)
