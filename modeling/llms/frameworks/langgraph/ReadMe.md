# LangGraph

## Summary

LangGraph is a low-level orchestration framework for building stateful, long-running agents using a graph-based architecture. Developed by LangChain Inc., it models agent workflows as directed graphs where nodes represent operations and edges define control flow. This explicit structure enables complex patterns like cycles, conditional branching, and human-in-the-loop interrupts that are difficult to achieve with linear chain-based approaches.

Key points to remember:

- Workflows are defined as StateGraphs with nodes (operations) and edges (transitions)
- State is explicitly typed and passed between nodes, enabling checkpointing and replay
- Conditional edges enable branching logic based on state or LLM decisions
- Built-in checkpointing allows agents to survive failures and support time-travel debugging
- Human-in-the-loop is first-class through interrupt mechanisms
- Integrates with LangSmith for observability and LangGraph Studio for visual development
- Can be used standalone or with LangChain components

## Installation and Setup

```bash
pip install -U langgraph
```

For full ecosystem integration:

```bash
pip install langgraph langchain langchain-openai langsmith
```

## Core Concepts

### StateGraph

The StateGraph is the central abstraction. It defines what data flows through the graph and how nodes can modify it:

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # Messages accumulate
    context: str                      # Context can be replaced
    iteration: int                    # Tracking loop count

# Create graph with typed state
graph = StateGraph(AgentState)
```

The `Annotated` type with a reducer function (like `add`) determines how state updates are merged. Without a reducer, new values replace old ones.

### Nodes

Nodes are functions that receive state and return partial state updates:

```python
def retrieve_context(state: AgentState) -> dict:
    """Retrieve relevant context for the query."""
    query = state["messages"][-1].content
    docs = retriever.search(query)
    return {"context": "\n".join(docs)}

def generate_response(state: AgentState) -> dict:
    """Generate a response using retrieved context."""
    messages = state["messages"]
    context = state["context"]

    response = llm.invoke([
        SystemMessage(content=f"Context: {context}"),
        *messages
    ])

    return {"messages": [response]}

def should_continue(state: AgentState) -> dict:
    """Increment iteration counter."""
    return {"iteration": state["iteration"] + 1}

# Add nodes to graph
graph.add_node("retrieve", retrieve_context)
graph.add_node("generate", generate_response)
graph.add_node("check", should_continue)
```

### Edges

Edges define control flow between nodes:

```python
# Simple edges - always transition
graph.add_edge(START, "retrieve")
graph.add_edge("retrieve", "generate")

# Conditional edges - route based on state
def route_decision(state: AgentState) -> str:
    """Decide next step based on state."""
    if state["iteration"] >= 3:
        return END
    if needs_more_context(state):
        return "retrieve"
    return END

graph.add_conditional_edges(
    "generate",
    route_decision,
    {
        "retrieve": "retrieve",  # Loop back
        END: END                   # Finish
    }
)

# Compile to executable
app = graph.compile()
```

### Executing the Graph

```python
# Invoke with initial state
result = app.invoke({
    "messages": [HumanMessage(content="What is RAG?")],
    "context": "",
    "iteration": 0
})

# Stream execution events
for event in app.stream(initial_state):
    print(f"Node: {event}")
```

## Common Patterns

### ReAct Agent

The classic reasoning-action loop implemented as a cycle:

```python
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search the web for information."""
    return search_api.query(query)

@tool
def calculate(expression: str) -> float:
    """Evaluate a mathematical expression."""
    return eval(expression)  # Simplified - use safe evaluation in production

# Prebuilt ReAct agent
llm = ChatOpenAI(model="gpt-4")
agent = create_react_agent(llm, tools=[search, calculate])

# Run agent
result = agent.invoke({
    "messages": [HumanMessage(content="What is 15% of the US population?")]
})
```

### Custom Tool-Calling Agent

Building the ReAct pattern explicitly for customization:

```python
from langchain_core.messages import AIMessage, ToolMessage

class ToolAgentState(TypedDict):
    messages: Annotated[list, add]

def call_model(state: ToolAgentState) -> dict:
    """Invoke the LLM with current messages."""
    response = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [response]}

def execute_tools(state: ToolAgentState) -> dict:
    """Execute any tool calls from the last message."""
    last_message = state["messages"][-1]
    tool_results = []

    for tool_call in last_message.tool_calls:
        tool = tool_map[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}

def should_continue(state: ToolAgentState) -> str:
    """Check if we should continue tool execution."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END

# Build graph
graph = StateGraph(ToolAgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", execute_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")  # Loop back after tool execution

agent = graph.compile()
```

### Multi-Agent Supervisor

Coordinating multiple specialized agents:

```python
class SupervisorState(TypedDict):
    messages: Annotated[list, add]
    next_agent: str
    final_response: str

def supervisor(state: SupervisorState) -> dict:
    """Route to appropriate agent based on task."""
    system = """You are a supervisor routing tasks to specialists:
    - researcher: for information gathering
    - coder: for code-related tasks
    - writer: for content creation

    Respond with the agent name or 'FINISH' if done."""

    response = llm.invoke([
        SystemMessage(content=system),
        *state["messages"]
    ])

    return {"next_agent": response.content.strip().lower()}

def researcher(state: SupervisorState) -> dict:
    """Research agent with search capabilities."""
    response = research_llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=f"[Researcher]: {response.content}")]}

def coder(state: SupervisorState) -> dict:
    """Coding agent with code execution."""
    response = code_llm.invoke(state["messages"])
    return {"messages": [AIMessage(content=f"[Coder]: {response.content}")]}

def route_to_agent(state: SupervisorState) -> str:
    """Route based on supervisor decision."""
    agent = state["next_agent"]
    if agent == "finish":
        return END
    return agent

# Build supervisor graph
graph = StateGraph(SupervisorState)
graph.add_node("supervisor", supervisor)
graph.add_node("researcher", researcher)
graph.add_node("coder", coder)

graph.add_edge(START, "supervisor")
graph.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {"researcher": "researcher", "coder": "coder", END: END}
)
graph.add_edge("researcher", "supervisor")
graph.add_edge("coder", "supervisor")

multi_agent = graph.compile()
```

## Checkpointing and Persistence

Checkpointing enables durability, human-in-the-loop, and time-travel debugging:

```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

# In-memory checkpointing (development)
memory = MemorySaver()

# SQLite checkpointing (production)
db_path = "checkpoints.db"
sqlite_saver = SqliteSaver.from_conn_string(db_path)

# Compile with checkpointer
agent = graph.compile(checkpointer=sqlite_saver)

# Execute with thread ID for persistence
config = {"configurable": {"thread_id": "user-123-session-456"}}

# First interaction
result1 = agent.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config
)

# Subsequent interaction continues from checkpoint
result2 = agent.invoke(
    {"messages": [HumanMessage(content="What did I just say?")]},
    config
)
```

### Time Travel

Replay from previous checkpoints:

```python
# Get all checkpoints for a thread
checkpoints = list(agent.get_state_history(config))

# Inspect a previous state
previous_state = checkpoints[2]
print(f"State at step 2: {previous_state.values}")

# Resume from a previous checkpoint
result = agent.invoke(
    {"messages": [HumanMessage(content="Different path")]},
    {"configurable": {"thread_id": "user-123", "checkpoint_id": previous_state.config["checkpoint_id"]}}
)
```

## Human-in-the-Loop

Interrupt execution for human oversight:

```python
from langgraph.prebuilt import ToolNode

# Define node that requires approval
def sensitive_action(state: AgentState) -> dict:
    """Action requiring human approval."""
    return {"pending_action": state["proposed_action"]}

# Compile with interrupt points
agent = graph.compile(
    checkpointer=memory,
    interrupt_before=["sensitive_action"]  # Pause before this node
)

# Execute until interrupt
config = {"configurable": {"thread_id": "approval-flow"}}
result = agent.invoke(initial_state, config)

# At this point, execution is paused
current_state = agent.get_state(config)
print(f"Pending: {current_state.values['proposed_action']}")

# Human reviews and approves
agent.update_state(
    config,
    {"approved": True},
    as_node="human_review"  # Attribute update to a virtual node
)

# Resume execution
final_result = agent.invoke(None, config)  # None continues from checkpoint
```

## Memory Systems

### Short-Term Memory

Conversation history within a session:

```python
from langgraph.prebuilt import MessagesState

class ConversationState(MessagesState):
    """State with built-in message handling."""
    summary: str  # Conversation summary for context compression

def summarize_if_needed(state: ConversationState) -> dict:
    """Compress long conversations."""
    if len(state["messages"]) > 20:
        summary = llm.invoke([
            SystemMessage(content="Summarize this conversation"),
            *state["messages"]
        ])
        return {
            "messages": state["messages"][-5:],  # Keep recent
            "summary": summary.content
        }
    return {}
```

### Long-Term Memory

Cross-session memory using external storage:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

class MemoryState(TypedDict):
    messages: Annotated[list, add]
    retrieved_memories: list

# Vector store for long-term memory
memory_store = Chroma(
    collection_name="agent_memories",
    embedding_function=OpenAIEmbeddings()
)

def retrieve_memories(state: MemoryState) -> dict:
    """Retrieve relevant past interactions."""
    query = state["messages"][-1].content
    memories = memory_store.similarity_search(query, k=3)
    return {"retrieved_memories": [m.page_content for m in memories]}

def store_memory(state: MemoryState) -> dict:
    """Store important interactions."""
    last_exchange = state["messages"][-2:]  # User + assistant
    memory_store.add_texts([str(last_exchange)])
    return {}
```

## Subgraphs and Modularity

Compose complex systems from smaller graphs:

```python
# Define a reusable subgraph
def create_research_subgraph():
    graph = StateGraph(ResearchState)
    graph.add_node("search", search_node)
    graph.add_node("summarize", summarize_node)
    graph.add_edge(START, "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", END)
    return graph.compile()

research_graph = create_research_subgraph()

# Use as node in parent graph
parent = StateGraph(ParentState)
parent.add_node("research", research_graph)  # Subgraph as node
parent.add_node("respond", respond_node)
parent.add_edge(START, "research")
parent.add_edge("research", "respond")
parent.add_edge("respond", END)
```

## Streaming

Stream execution for responsive UIs:

```python
# Stream all events
async for event in agent.astream_events(
    {"messages": [HumanMessage(content="Tell me a story")]},
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        # Token-level streaming from LLM
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_tool_start":
        # Tool execution started
        print(f"\nUsing tool: {event['name']}")
    elif event["event"] == "on_chain_end":
        # Node completed
        print(f"\nCompleted: {event['name']}")
```

## Debugging with LangSmith

Integrate with LangSmith for production observability:

```python
import os

# Enable tracing
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-agent-project"

# Traces automatically captured
result = agent.invoke(initial_state)

# Access traces programmatically
from langsmith import Client
client = Client()
runs = client.list_runs(project_name="my-agent-project")
```

## Production Deployment

### LangGraph Platform

Deploy as a managed service:

```python
# langgraph.json configuration
{
    "graphs": {
        "my_agent": "./agent.py:agent"
    },
    "dependencies": ["langchain", "langchain-openai"]
}
```

### Self-Hosted

Deploy as a standard Python application:

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()

# Expose agent as API
add_routes(app, agent, path="/agent")

# Custom endpoint with streaming
@app.post("/chat")
async def chat(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    async def stream_response():
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config,
            version="v2"
        ):
            yield event

    return StreamingResponse(stream_response())
```

## Comparison with Strands

| Aspect | LangGraph | Strands |
|--------|-----------|---------|
| Workflow Definition | Explicit graph structure | Implicit through model reasoning |
| State Management | TypedDict with reducers | Session-based |
| Learning Curve | Steeper | Gentler |
| Customization | Very high | Moderate |
| Debugging | Visual graph traces | Log-based traces |
| Best For | Complex, deterministic workflows | Rapid development, conversational agents |

Choose LangGraph when you need explicit control over execution flow, complex branching logic, or robust checkpointing for long-running workflows. The graph-based approach provides clarity for systems where the execution path matters as much as the outcome.

## Further Reading

- LangGraph Documentation: https://docs.langchain.com/oss/python/langgraph/overview
- LangGraph GitHub: https://github.com/langchain-ai/langgraph
- LangChain Academy: Free LangGraph courses
- LangGraph Studio: Visual development environment
