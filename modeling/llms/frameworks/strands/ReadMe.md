# Strands Agents

## Summary

Strands Agents is an open-source SDK from AWS that takes a model-driven approach to building AI agents. Rather than requiring developers to define explicit workflows, Strands leverages the planning and reasoning capabilities of modern LLMs to orchestrate tool usage and task completion. This design philosophy enables building production-ready agents in just a few lines of code while remaining flexible enough for complex multi-agent systems.

Key points to remember:

- Model-driven architecture lets the LLM decide tool execution order and reasoning flow
- Supports multiple model providers including Bedrock, Anthropic, OpenAI, and Ollama
- Native Model Context Protocol support for accessing thousands of pre-built tools
- Simple decorator-based tool definition with automatic schema generation
- Session management for conversation persistence across interactions
- Agent-to-Agent protocol for multi-agent orchestration
- Production-tested by AWS teams including Amazon Q Developer and AWS Glue
- Apache 2.0 licensed with Python and TypeScript implementations

## Installation and Setup

Requires Python 3.10 or higher:

```bash
pip install strands-agents strands-agents-tools
```

For specific model providers:

```bash
# AWS Bedrock (default)
pip install strands-agents[bedrock]

# Anthropic direct
pip install strands-agents[anthropic]

# OpenAI
pip install strands-agents[openai]
```

## Basic Usage

The simplest agent requires just a few lines:

```python
from strands import Agent
from strands_tools import calculator

# Create agent with a tool
agent = Agent(tools=[calculator])

# Run the agent
response = agent("What is the square root of 1764?")
print(response)
```

The agent automatically:
1. Receives the user query
2. Decides the calculator tool is needed
3. Calls the tool with appropriate arguments
4. Incorporates the result into its response

## Core Concepts

### The Agent Loop

Strands implements an agentic loop where the model iteratively reasons and acts:

```python
from strands import Agent

class Agent:
    def __call__(self, prompt: str) -> str:
        """
        Simplified view of the agent loop:
        1. Send prompt + tools to model
        2. Model returns either response or tool calls
        3. If tool calls, execute and feed results back
        4. Repeat until model returns final response
        """
        messages = [{"role": "user", "content": prompt}]

        while True:
            response = self.model.invoke(messages, tools=self.tools)

            if response.is_final:
                return response.content

            # Execute tool calls
            for tool_call in response.tool_calls:
                result = self.execute_tool(tool_call)
                messages.append(tool_result_message(result))
```

### Model Configuration

Strands supports multiple model providers through a consistent interface:

```python
from strands import Agent
from strands.models import BedrockModel, AnthropicModel, OpenAIModel

# AWS Bedrock (default)
bedrock_model = BedrockModel(
    model_id="us.anthropic.claude-sonnet-4-20250514",
    temperature=0.3,
    max_tokens=4096
)

# Anthropic direct API
anthropic_model = AnthropicModel(
    model_id="claude-sonnet-4-20250514",
    api_key="your-api-key"
)

# OpenAI
openai_model = OpenAIModel(
    model_id="gpt-4o",
    api_key="your-api-key"
)

# Create agent with specific model
agent = Agent(model=bedrock_model, tools=[...])
```

### System Prompts

Customize agent behavior with system prompts:

```python
agent = Agent(
    system_prompt="""You are a helpful data analyst assistant.
    You have access to tools for querying databases and creating visualizations.
    Always explain your reasoning before executing queries.
    Format numerical results with appropriate precision.""",
    tools=[query_database, create_chart]
)
```

## Tool Definition

### Decorator-Based Tools

The simplest way to create tools is with the `@tool` decorator:

```python
from strands import Agent, tool

@tool
def get_weather(city: str, units: str = "celsius") -> dict:
    """Get current weather for a city.

    Args:
        city: The city name to get weather for
        units: Temperature units, either 'celsius' or 'fahrenheit'

    Returns:
        Weather data including temperature, conditions, and humidity
    """
    # Implementation
    response = weather_api.get(city)
    temp = response["temp_c"] if units == "celsius" else response["temp_f"]
    return {
        "city": city,
        "temperature": temp,
        "units": units,
        "conditions": response["conditions"],
        "humidity": response["humidity"]
    }

@tool
def search_products(query: str, max_results: int = 10) -> list:
    """Search the product catalog.

    Args:
        query: Search terms
        max_results: Maximum number of results to return

    Returns:
        List of matching products with names and prices
    """
    return product_db.search(query, limit=max_results)

agent = Agent(tools=[get_weather, search_products])
```

The decorator:
- Extracts the function signature for the tool schema
- Uses the docstring as the tool description for the LLM
- Handles argument parsing and validation automatically

### Tool Return Types

Tools can return various types:

```python
@tool
def simple_tool(x: int) -> str:
    """Returns a string."""
    return f"Result: {x}"

@tool
def dict_tool(query: str) -> dict:
    """Returns structured data."""
    return {"query": query, "results": [...]}

@tool
def list_tool(category: str) -> list:
    """Returns a list of items."""
    return ["item1", "item2", "item3"]
```

### Async Tools

For I/O-bound operations:

```python
import asyncio
from strands import tool

@tool
async def fetch_data(url: str) -> dict:
    """Fetch data from an API endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

## Model Context Protocol Integration

Strands has native MCP support for accessing external tool servers:

```python
from strands import Agent
from strands.mcp import MCPClient

# Connect to MCP server
filesystem_tools = MCPClient("npx -y @modelcontextprotocol/server-filesystem /tmp")

# Use MCP tools alongside custom tools
agent = Agent(tools=[
    *filesystem_tools.get_tools(),
    custom_tool
])

# Agent can now read/write files through MCP
response = agent("Create a file called notes.txt with today's date")
```

### Multiple MCP Servers

```python
# Connect to multiple servers
filesystem = MCPClient("npx -y @modelcontextprotocol/server-filesystem /workspace")
github = MCPClient("npx -y @modelcontextprotocol/server-github")
slack = MCPClient("npx -y @modelcontextprotocol/server-slack")

agent = Agent(tools=[
    *filesystem.get_tools(),
    *github.get_tools(),
    *slack.get_tools()
])
```

## Sessions and Memory

Sessions enable conversation persistence:

```python
from strands import Agent, Session

# Create a session for persistence
session = Session(session_id="user-123-conversation")

agent = Agent(
    tools=[...],
    session=session
)

# First interaction
agent("My name is Alice")

# Later interaction - agent remembers context
agent("What's my name?")  # Agent recalls "Alice"

# Session can be persisted to storage
session.save("sessions/user-123.json")

# And restored later
restored_session = Session.load("sessions/user-123.json")
```

### Session Configuration

```python
from strands import Session

session = Session(
    session_id="unique-id",
    max_messages=100,        # Limit conversation history
    summary_threshold=50,    # Summarize after N messages
    metadata={               # Custom metadata
        "user_id": "123",
        "started_at": datetime.now()
    }
)
```

## Multi-Agent Systems

### Agents as Tools

Use specialized agents as tools for a coordinator:

```python
from strands import Agent, tool

# Specialized agents
researcher = Agent(
    system_prompt="You are a research specialist...",
    tools=[web_search, arxiv_search]
)

coder = Agent(
    system_prompt="You are a coding specialist...",
    tools=[code_executor, file_writer]
)

# Wrap agents as tools
@tool
def research(query: str) -> str:
    """Delegate research tasks to a specialist."""
    return researcher(query)

@tool
def write_code(spec: str) -> str:
    """Delegate coding tasks to a specialist."""
    return coder(spec)

# Coordinator agent
coordinator = Agent(
    system_prompt="You coordinate between specialists...",
    tools=[research, write_code]
)

# Coordinator decides which specialist to use
coordinator("Research transformer architectures and write a summary")
```

### Agent-to-Agent Protocol

For more sophisticated multi-agent communication:

```python
from strands import Agent
from strands.multi_agent import AgentNetwork, A2AProtocol

# Define agents
agents = {
    "planner": Agent(system_prompt="You create plans..."),
    "executor": Agent(system_prompt="You execute tasks..."),
    "reviewer": Agent(system_prompt="You review outputs...")
}

# Create network with A2A protocol
network = AgentNetwork(
    agents=agents,
    protocol=A2AProtocol()
)

# Agents can communicate through the network
result = network.run(
    task="Build a web scraper",
    entry_agent="planner"
)
```

### Swarm Pattern

For autonomous agent collaboration:

```python
from strands.multi_agent import Swarm

swarm = Swarm(
    agents=[agent1, agent2, agent3],
    strategy="collaborative",  # or "competitive", "sequential"
    max_rounds=10
)

# Agents collaborate on complex task
result = swarm.run("Analyze this dataset and create a report")
```

## Streaming

Stream responses for responsive interfaces:

```python
from strands import Agent

agent = Agent(tools=[...], streaming=True)

# Callback-based streaming
def on_token(token: str):
    print(token, end="", flush=True)

def on_tool_call(tool_name: str, args: dict):
    print(f"\n[Using {tool_name}...]")

agent(
    "Explain quantum computing",
    on_token=on_token,
    on_tool_call=on_tool_call
)

# Or iterate over stream
for event in agent.stream("Tell me a story"):
    if event.type == "token":
        print(event.content, end="")
    elif event.type == "tool_start":
        print(f"\n[{event.tool_name}]")
    elif event.type == "tool_result":
        print(f"[Result: {event.result}]")
```

## Guardrails and Safety

### Input Validation

```python
from strands import Agent
from strands.guardrails import InputGuard

# Block certain topics
input_guard = InputGuard(
    blocked_patterns=[
        r"password|secret|credential",
        r"hack|exploit|attack"
    ],
    max_length=10000
)

agent = Agent(
    tools=[...],
    input_guard=input_guard
)
```

### Output Filtering

```python
from strands.guardrails import OutputGuard, PIIRedactor

# Redact PII from outputs
pii_redactor = PIIRedactor(
    redact_types=["email", "phone", "ssn", "credit_card"]
)

output_guard = OutputGuard(
    filters=[pii_redactor],
    blocked_patterns=[r"internal-only"]
)

agent = Agent(
    tools=[...],
    output_guard=output_guard
)
```

### AWS Bedrock Guardrails

```python
from strands.models import BedrockModel

model = BedrockModel(
    model_id="anthropic.claude-sonnet-4-20250514-v1:0",
    guardrail_id="your-guardrail-id",
    guardrail_version="1"
)
```

## Observability

### Tracing

```python
from strands import Agent
from strands.telemetry import Tracer

# Enable tracing
tracer = Tracer(
    service_name="my-agent",
    exporter="otlp",  # OpenTelemetry
    endpoint="http://localhost:4317"
)

agent = Agent(tools=[...], tracer=tracer)

# Traces include:
# - Agent invocations
# - Model calls with latency
# - Tool executions
# - Token usage
```

### Logging

```python
import logging
from strands import Agent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("strands")

# Detailed agent logs
agent = Agent(tools=[...], verbose=True)
```

### Metrics

```python
from strands.telemetry import Metrics

metrics = Metrics(
    namespace="my-agent",
    dimensions={"environment": "production"}
)

agent = Agent(tools=[...], metrics=metrics)

# Metrics collected:
# - invocation_count
# - latency_ms
# - tool_calls
# - token_usage
# - error_count
```

## Error Handling

### Tool Errors

```python
@tool
def risky_operation(data: str) -> str:
    """Operation that might fail."""
    try:
        result = external_api.process(data)
        return result
    except APIError as e:
        # Return error message - agent can retry or adjust
        return f"Error: {e.message}. Try with different parameters."
```

### Agent-Level Error Handling

```python
from strands import Agent
from strands.errors import ToolExecutionError, ModelError

agent = Agent(tools=[...])

try:
    result = agent("Complex task")
except ToolExecutionError as e:
    print(f"Tool {e.tool_name} failed: {e.message}")
    # Handle tool failure
except ModelError as e:
    print(f"Model error: {e.message}")
    # Handle model API issues
except Exception as e:
    print(f"Unexpected error: {e}")
```

### Retry Configuration

```python
from strands import Agent

agent = Agent(
    tools=[...],
    max_retries=3,
    retry_delay=1.0,  # seconds
    retry_on=["rate_limit", "timeout"]
)
```

## Production Deployment

### AWS Lambda

```python
from strands import Agent

# Initialize outside handler for reuse
agent = Agent(
    model=BedrockModel(model_id="anthropic.claude-sonnet-4-20250514-v1:0"),
    tools=[...]
)

def lambda_handler(event, context):
    user_message = event["message"]
    session_id = event.get("session_id", "default")

    response = agent(
        user_message,
        session_id=session_id
    )

    return {
        "statusCode": 200,
        "body": {"response": response}
    }
```

### FastAPI

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from strands import Agent

app = FastAPI()
agent = Agent(tools=[...], streaming=True)

@app.post("/chat")
async def chat(message: str, session_id: str = "default"):
    async def generate():
        for event in agent.stream(message, session_id=session_id):
            yield f"data: {event.json()}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

## Comparison with LangGraph

| Aspect | Strands | LangGraph |
|--------|---------|-----------|
| Philosophy | Model-driven | Graph-defined |
| Setup Complexity | Minimal | Moderate |
| Workflow Control | Implicit | Explicit |
| State Management | Sessions | StateGraph |
| Multi-Agent | A2A protocol, swarms | Subgraphs |
| Best For | Rapid development | Complex workflows |
| AWS Integration | Native | Via providers |

Choose Strands when:
- Rapid prototyping is priority
- Building conversational agents
- Leveraging AWS services extensively
- Preferring minimal boilerplate
- Team is new to agent development

Choose LangGraph when:
- Complex branching workflows needed
- Explicit execution control required
- Time-travel debugging valuable
- Building deterministic pipelines

## Further Reading

- Strands Documentation: https://strandsagents.com/latest/
- GitHub Repository: https://github.com/strands-agents/sdk-python
- AWS Blog: https://aws.amazon.com/blogs/opensource/introducing-strands-agents-an-open-source-ai-agents-sdk/
- Tools Package: https://pypi.org/project/strands-agents-tools/
