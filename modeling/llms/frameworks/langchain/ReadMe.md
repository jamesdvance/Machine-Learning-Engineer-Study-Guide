# LangChain

## Summary

LangChain is an open-source framework that provides abstractions and integrations for building LLM-powered applications. Originally focused on chaining together LLM calls with other components, it has evolved into a foundation layer that standardizes model interactions, tool integration, and agent construction. LangChain agents are built on top of LangGraph, inheriting production features like persistence and human-in-the-loop while providing a simpler API for common patterns.

Key points to remember:

- Provides a standardized interface across hundreds of LLM providers, preventing vendor lock-in
- The @tool decorator creates callable functions that agents can invoke based on conversation context
- Model initialization abstracts provider-specific APIs into a consistent interface
- Agents combine models, tools, and system prompts to create autonomous decision-making systems
- Supports structured output via Pydantic models, TypedDict, and JSON Schema
- Memory enables agents to maintain state across interactions via checkpointers
- Integration with LangSmith provides observability through execution traces and debugging tools
- Built on LangGraph for durability, streaming, and advanced workflow patterns

## Installation and Setup

LangChain requires Python 3.10 or higher:

```bash
pip install -U langchain
```

Install provider packages as needed:

```bash
# OpenAI integration
pip install -U langchain-openai

# Anthropic integration
pip install -U langchain-anthropic

# Google integration
pip install -U langchain-google-genai

# AWS Bedrock
pip install -U langchain-aws
```

Set provider credentials via environment variables:

```bash
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"
```

## Core Concepts

### Models

LangChain abstracts LLM provider APIs into a consistent interface. The primary entry point is init_chat_model():

```python
from langchain.chat_models import init_chat_model

# OpenAI
model = init_chat_model("gpt-4o", temperature=0.7)

# Anthropic
model = init_chat_model(
    "claude-sonnet-4-20250514",
    model_provider="anthropic"
)

# Google
model = init_chat_model(
    "gemini-1.5-pro",
    model_provider="google_genai"
)
```

This abstraction enables switching providers without changing application logic. The same code works across different LLMs:

```python
from langchain_core.messages import HumanMessage, SystemMessage

messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="Explain quantum computing briefly.")
]

response = model.invoke(messages)
print(response.content)
```

### Messages

LangChain uses a message-based conversation model:

```python
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage
)

conversation = [
    SystemMessage(content="You are a data analyst."),
    HumanMessage(content="What's the average of [1, 2, 3, 4, 5]?"),
    AIMessage(content="The average is 3."),
    HumanMessage(content="Now calculate the standard deviation.")
]

response = model.invoke(conversation)
```

ToolMessage is used to return results from tool executions back to the model:

```python
tool_result = ToolMessage(
    content="Search found 15 results for 'AI trends'",
    tool_call_id="call_abc123"
)
```

### Tools

Tools are callable functions that agents can invoke. The @tool decorator is the primary creation method:

```python
from langchain.tools import tool

@tool
def search_database(query: str, limit: int = 10) -> str:
    """Search the customer database for records matching the query.

    Args:
        query: The search terms to match
        limit: Maximum number of results to return

    Returns:
        A formatted string of matching records
    """
    results = db.search(query, limit=limit)
    return format_results(results)
```

Type hints are required and define the tool's input schema. The docstring becomes the description that helps the model decide when to use the tool. Clear, specific descriptions improve tool selection accuracy.

Customize tool metadata:

```python
@tool("customer_search")  # Custom name
def search_database(query: str) -> str:
    """Search for customer records."""
    return db.search(query)

# Or with explicit description
@tool(description="Search the customer database by name, email, or ID")
def search_database(query: str) -> str:
    return db.search(query)
```

For complex inputs, use Pydantic models:

```python
from pydantic import BaseModel, Field

class SearchParams(BaseModel):
    query: str = Field(description="Search terms")
    filters: dict = Field(default={}, description="Optional filters")
    limit: int = Field(default=10, ge=1, le=100)

@tool(args_schema=SearchParams)
def advanced_search(query: str, filters: dict, limit: int) -> str:
    """Perform an advanced search with filtering."""
    return search_engine.query(query, filters=filters, limit=limit)
```

Tools can access runtime context for dynamic behavior:

```python
from langchain.tools import tool

@tool
def get_user_preferences(runtime) -> dict:
    """Get the current user's preferences."""
    user_id = runtime.context.get("user_id")
    return preferences_db.get(user_id)
```

### Agents

Agents combine models, tools, and system prompts into autonomous systems:

```python
from langchain.agents import create_agent
from langchain.tools import tool

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return weather_api.get(city)

@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    return search_api.query(query)

agent = create_agent(
    model="claude-sonnet-4-20250514",
    tools=[get_weather, search_web],
    system_prompt="""You are a helpful assistant that can look up
    weather and search the web. Always cite your sources."""
)

response = agent.invoke("What's the weather in Tokyo?")
print(response["messages"][-1].content)
```

The agent loop works as follows:

1. The model receives the conversation and available tools
2. The model decides whether to respond directly or call a tool
3. If a tool is called, the result is added to the conversation
4. The loop continues until the model provides a final response

### Structured Output

Force responses into specific formats:

```python
from pydantic import BaseModel
from typing import List

class MovieReview(BaseModel):
    title: str
    rating: float
    summary: str
    pros: List[str]
    cons: List[str]

structured_model = model.with_structured_output(MovieReview)

review = structured_model.invoke(
    "Give me a review of The Matrix"
)
print(f"{review.title}: {review.rating}/10")
```

Structured output works with TypedDict and JSON Schema as well:

```python
from typing import TypedDict

class Review(TypedDict):
    title: str
    rating: float
    summary: str

structured_model = model.with_structured_output(Review)
```

## Memory and State

Agents can maintain conversation state across interactions using checkpointers:

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import MemorySaver

# In-memory checkpointer for development
memory = MemorySaver()

agent = create_agent(
    model="claude-sonnet-4-20250514",
    tools=[...],
    system_prompt="You are a helpful assistant.",
    checkpointer=memory
)

# First interaction
config = {"configurable": {"thread_id": "user-123"}}
response1 = agent.invoke(
    {"messages": [HumanMessage(content="My name is Alice")]},
    config
)

# Later interaction - agent remembers
response2 = agent.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config
)
# Agent responds with "Alice"
```

For production, use persistent checkpointers:

```python
from langgraph.checkpoint.sqlite import SqliteSaver

db_path = "checkpoints.db"
checkpointer = SqliteSaver.from_conn_string(db_path)

agent = create_agent(
    model="...",
    tools=[...],
    checkpointer=checkpointer
)
```

## Streaming

Stream responses for responsive interfaces:

```python
# Stream complete events
for event in agent.stream(
    {"messages": [HumanMessage(content="Tell me a story")]},
    config
):
    print(event)

# Async streaming
async for event in agent.astream_events(
    {"messages": [HumanMessage(content="Tell me a story")]},
    config,
    version="v2"
):
    if event["event"] == "on_chat_model_stream":
        # Token-level streaming
        print(event["data"]["chunk"].content, end="")
    elif event["event"] == "on_tool_start":
        print(f"\nUsing tool: {event['name']}")
```

## LangChain Expression Language (LCEL)

LCEL provides a declarative way to compose components using the pipe operator:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("user", "{input}")
])

chain = prompt | model | StrOutputParser()

result = chain.invoke({"input": "Explain machine learning"})
```

The Runnable interface that powers LCEL provides consistent methods across components:

- invoke(): Synchronous execution
- ainvoke(): Async execution
- stream(): Stream output chunks
- batch(): Process multiple inputs

Compose more complex chains:

```python
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

answer = rag_chain.invoke("What is RAG?")
```

## Integration Patterns

### Multiple Model Providers

Switch providers without code changes:

```python
from langchain.chat_models import init_chat_model
import os

def get_model(provider: str):
    """Factory for creating models from different providers."""
    configs = {
        "openai": ("gpt-4o", "openai"),
        "anthropic": ("claude-sonnet-4-20250514", "anthropic"),
        "google": ("gemini-1.5-pro", "google_genai")
    }

    model_name, model_provider = configs[provider]
    return init_chat_model(model_name, model_provider=model_provider)

# Use environment variable to select provider
model = get_model(os.getenv("LLM_PROVIDER", "openai"))
```

### External Tool Integrations

LangChain provides hundreds of pre-built tool integrations:

```python
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

search = DuckDuckGoSearchRun()
wikipedia = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper())

agent = create_agent(
    model="claude-sonnet-4-20250514",
    tools=[search, wikipedia],
    system_prompt="You are a research assistant."
)
```

### Retrieval-Augmented Generation

Combine retrieval with generation:

```python
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.tools.retriever import create_retriever_tool

# Create vector store
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings()
)

# Create retriever tool
retriever_tool = create_retriever_tool(
    vectorstore.as_retriever(),
    name="knowledge_base",
    description="Search the knowledge base for relevant information"
)

agent = create_agent(
    model="claude-sonnet-4-20250514",
    tools=[retriever_tool],
    system_prompt="Use the knowledge base to answer questions."
)
```

## Observability with LangSmith

LangSmith provides tracing and debugging for LangChain applications:

```python
import os

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "your-api-key"
os.environ["LANGCHAIN_PROJECT"] = "my-project"

# All agent executions are now traced
result = agent.invoke({"messages": [HumanMessage(content="Hello")]})
```

Traces capture:

- Model invocations with inputs and outputs
- Tool calls and their results
- Token usage and latency metrics
- Error states and stack traces

Access traces programmatically:

```python
from langsmith import Client

client = Client()
runs = client.list_runs(
    project_name="my-project",
    filter="eq(status, 'error')"  # Find failed runs
)
```

## Error Handling

Handle common failure modes:

```python
from langchain_core.exceptions import OutputParserException
from langchain.callbacks import get_openai_callback

try:
    with get_openai_callback() as cb:
        result = agent.invoke({"messages": messages})
        print(f"Tokens used: {cb.total_tokens}")
except OutputParserException as e:
    # Handle parsing failures
    logger.error(f"Failed to parse output: {e}")
except Exception as e:
    # Handle API errors, rate limits, etc.
    logger.error(f"Agent execution failed: {e}")
```

Implement retry logic:

```python
from langchain_core.runnables import RunnableConfig

config = RunnableConfig(
    max_concurrency=5,
    recursion_limit=25,
    tags=["production"]
)

result = agent.invoke(messages, config)
```

## LangChain vs LangGraph

LangChain agents are built on LangGraph but provide a simpler API. Understanding when to use each:

Use LangChain when:

- Building standard agent patterns quickly
- Prototyping agent capabilities
- Team is new to agent development
- Default agent loop is sufficient

Use LangGraph directly when:

- Complex branching workflows are needed
- Explicit control over execution is required
- Custom state management beyond messages
- Deterministic-agentic hybrid workflows

```python
# LangChain: Simple agent API
agent = create_agent(model, tools, system_prompt)

# LangGraph: Explicit graph construction
from langgraph.graph import StateGraph, START, END

graph = StateGraph(AgentState)
graph.add_node("agent", call_model)
graph.add_node("tools", execute_tools)
graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue)
graph.add_edge("tools", "agent")
app = graph.compile()
```

## Comparison with Other Frameworks

| Aspect | LangChain | CrewAI | Strands |
|--------|-----------|--------|---------|
| Primary Focus | Foundation library | Multi-agent teams | Rapid development |
| Agent Definition | Model + tools + prompt | Role/goal/backstory | Minimal configuration |
| Orchestration | Via LangGraph | Crews with processes | Model-driven |
| Tool Ecosystem | Hundreds of integrations | 40+ built-in | MCP + custom |
| Configuration | Code-based | YAML + decorators | Code-based |
| Best for | General LLM applications | Team workflows | Quick prototyping |

LangChain serves as a foundation layer that other frameworks build upon. It provides the model abstractions, tool patterns, and integration ecosystem that higher-level frameworks leverage.

## Production Deployment

### FastAPI Integration

```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI()

# Expose agent as REST API
add_routes(app, agent, path="/agent")

# Custom endpoint with streaming
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(message: str, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}

    async def stream():
        async for event in agent.astream_events(
            {"messages": [HumanMessage(content=message)]},
            config,
            version="v2"
        ):
            yield f"data: {event}\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")
```

### Configuration Best Practices

```python
from pydantic_settings import BaseSettings

class AgentConfig(BaseSettings):
    model_name: str = "claude-sonnet-4-20250514"
    model_provider: str = "anthropic"
    temperature: float = 0.7
    max_tokens: int = 4096
    checkpoint_db: str = "checkpoints.db"

    class Config:
        env_prefix = "AGENT_"

config = AgentConfig()
model = init_chat_model(
    config.model_name,
    model_provider=config.model_provider,
    temperature=config.temperature,
    max_tokens=config.max_tokens
)
```

## Further Reading

- LangChain Documentation: https://docs.langchain.com
- LangChain Python API: https://api.python.langchain.com
- GitHub Repository: https://github.com/langchain-ai/langchain
- LangSmith: https://smith.langchain.com
- LangGraph Documentation: https://docs.langchain.com/langgraph
