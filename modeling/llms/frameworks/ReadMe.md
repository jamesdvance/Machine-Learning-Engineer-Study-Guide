# LLM Agent Frameworks

## Summary

LLM agent frameworks provide structured abstractions for building autonomous AI systems that can reason, plan, and take actions. Rather than building agent loops, tool integrations, and state management from scratch, these frameworks offer production-tested primitives that accelerate development while handling edge cases encountered in real-world deployments.

Key points to remember:

- Agent frameworks abstract the agent loop, tool execution, memory, and state management
- Graph-based frameworks like LangGraph excel at complex, branching workflows with checkpointing
- Model-driven frameworks like Strands prioritize simplicity by leveraging LLM capabilities for orchestration
- Role-based frameworks like CrewAI model agents as team members with specialized expertise
- Foundation libraries like LangChain provide standardized abstractions across LLM providers
- Framework choice depends on workflow complexity, team expertise, and production requirements
- All major frameworks support Model Context Protocol for standardized tool integration
- Production features like human-in-the-loop, streaming, and observability differentiate mature frameworks

## Framework Landscape

The LLM agent framework ecosystem has matured significantly, with frameworks generally falling into two architectural philosophies:

### Graph-Based Orchestration

Frameworks like LangGraph model agent workflows as directed graphs where nodes represent operations and edges define control flow. This approach provides explicit visibility into execution paths and enables sophisticated patterns like cycles, branching, and conditional routing.

Advantages:
- Fine-grained control over execution flow
- Built-in support for complex state machines
- Visualization and debugging through graph representations
- Checkpointing at arbitrary points for durability

Trade-offs:
- Higher learning curve for simple use cases
- More boilerplate for straightforward agents
- Graph structure must be defined upfront

### Model-Driven Orchestration

Frameworks like Strands Agents embrace the capabilities of modern LLMs to handle orchestration decisions. Rather than explicitly defining workflows, developers provide tools and let the model determine execution order through its reasoning capabilities.

Advantages:
- Minimal boilerplate for common patterns
- Leverages LLM improvements automatically
- Faster prototyping and iteration
- Natural fit for conversational agents

Trade-offs:
- Less explicit control over execution
- Behavior depends on model capabilities
- Harder to enforce strict execution orders

### Role-Based Multi-Agent Systems

Frameworks like CrewAI organize agents as teams with defined roles, goals, and areas of expertise. This mirrors organizational structures where specialists collaborate on complex projects.

Advantages:
- Natural mapping to team-based workflows
- Agents develop specialized expertise
- Built-in delegation and collaboration patterns
- YAML configuration separates concerns

Trade-offs:
- More overhead for single-agent use cases
- Role definitions require upfront design
- Less flexible for highly dynamic workflows

### Foundation Libraries

LangChain provides the foundational abstractions that many frameworks build upon, offering standardized interfaces for models, tools, and memory without prescribing a specific orchestration pattern.

Advantages:
- Hundreds of model and tool integrations
- Consistent API across providers
- Avoids vendor lock-in
- Flexible composition patterns

Trade-offs:
- Less opinionated about agent architecture
- Requires more decisions from developers
- Advanced patterns require LangGraph knowledge

## Choosing a Framework

| Factor | LangGraph | Strands | CrewAI | LangChain |
|--------|-----------|---------|--------|-----------|
| Workflow Complexity | Complex, multi-branch | Linear to moderate | Team-based | Varies |
| Control Requirements | Explicit, deterministic | Flexible, adaptive | Role-defined | Composable |
| Development Speed | Moderate | Fast | Moderate | Fast |
| Multi-Agent | Subgraphs | A2A protocol | Native crews | Via LangGraph |
| Debugging | Graph visualization | Trace logging | Multiple integrations | LangSmith |
| Team Background | State machines, workflows | Rapid prototyping | Team organization | General LLM apps |
| Production Maturity | Extensive | Growing | Growing | Extensive |

### When to Use Graph-Based Frameworks

- Multi-step approval workflows with human checkpoints
- Pipelines requiring guaranteed execution order
- Systems with complex error handling and recovery
- Applications needing time-travel debugging
- Long-running agents that must survive restarts

### When to Use Model-Driven Frameworks

- Conversational agents with tool access
- Rapid prototyping of agent capabilities
- Teams prioritizing development velocity
- Use cases where LLM reasoning determines workflow
- Integration-heavy agents using many external tools

### When to Use Role-Based Frameworks

- Workflows that mirror organizational team structures
- Agents requiring distinct specialized expertise
- Projects benefiting from YAML-based configuration
- Multi-agent collaboration with delegation patterns
- Enterprise deployments with clear role boundaries

### When to Use Foundation Libraries

- Building custom agent architectures
- Needing broad provider and integration support
- Avoiding vendor lock-in across LLM providers
- Combining with other frameworks (e.g., LangGraph for orchestration)
- Rapid prototyping before committing to a specific pattern

## Common Framework Capabilities

### State Management

All production frameworks provide mechanisms for maintaining state across agent interactions:

```python
# State persists across conversation turns
# Frameworks differ in how state is defined and accessed
state = {
    "messages": [],           # Conversation history
    "context": {},            # Working memory
    "tool_results": [],       # Results from tool calls
    "metadata": {}            # Session information
}
```

### Tool Integration

Modern frameworks support multiple tool integration patterns:

```python
# Decorator-based tool definition (common pattern)
@tool
def search_database(query: str) -> list:
    """Search the product database for items matching the query."""
    return db.search(query)

# MCP server integration (standardized protocol)
mcp_tools = MCPClient("npx -y @modelcontextprotocol/server-filesystem")
agent = Agent(tools=[search_database, *mcp_tools])
```

### Streaming and Observability

Production agents require visibility into execution:

```python
# Stream agent responses and tool calls
for event in agent.stream(task):
    if event.type == "thinking":
        log_reasoning(event.content)
    elif event.type == "tool_call":
        log_tool_usage(event.tool, event.args)
    elif event.type == "response":
        yield event.content

# Trace integration for debugging
with tracer.span("agent_execution"):
    result = agent.run(task)
```

### Human-in-the-Loop

Incorporating human oversight at critical decision points:

```python
# Interrupt execution for human approval
if action.requires_approval:
    checkpoint = agent.pause()
    approval = await get_human_approval(action)
    if approval:
        agent.resume(checkpoint)
    else:
        agent.cancel()
```

## Framework Comparison

| Feature | LangGraph | Strands | CrewAI | LangChain |
|---------|-----------|---------|--------|-----------|
| Architecture | Graph-based | Model-driven | Role-based teams | Foundation library |
| State Management | Explicit StateGraph | Session-based | Memory types | Via checkpointers |
| Checkpointing | Built-in | Via sessions | Flow persistence | Via LangGraph |
| Multi-Agent | Subgraphs, orchestration | A2A protocol, swarms | Crews with delegation | Via LangGraph |
| MCP Support | Via tools | Native | Via tools | Via tools |
| Streaming | Full support | Full support | Full support | Full support |
| Human-in-the-Loop | First-class | Supported | Via Flows | Via LangGraph |
| Debugging | LangSmith integration | Trace logging | Multiple integrations | LangSmith |
| Configuration | Code-based | Code-based | YAML + decorators | Code-based |
| License | MIT | Apache 2.0 | MIT | MIT |
| Primary Maintainer | LangChain | AWS | CrewAI Inc | LangChain |

## Production Considerations

### Durability and Recovery

Production agents must handle failures gracefully:

- Checkpoint state before expensive operations
- Implement idempotent tool calls where possible
- Design for agent restarts without data loss
- Consider timeout handling for long-running tasks

### Security

Agent frameworks execute code and access external systems:

- Validate tool inputs before execution
- Implement permission systems for sensitive operations
- Sandbox code execution tools
- Audit log all agent actions

### Cost Management

Agent execution can become expensive:

- Monitor token usage per conversation
- Implement circuit breakers for runaway loops
- Cache tool results when appropriate
- Set maximum iteration limits

## Framework Ecosystem

Beyond the frameworks covered in detail, the broader ecosystem includes:

- **AutoGen** (Microsoft): Multi-agent conversation framework
- **Semantic Kernel** (Microsoft): Enterprise-focused agent framework
- **Haystack**: Pipeline-based framework with agent support
- **DSPy**: Programmatic prompt optimization with agent patterns

Each framework makes different trade-offs between flexibility, simplicity, and production-readiness. The detailed chapters that follow cover the major frameworks representing different architectural approaches: graph-based orchestration (LangGraph), model-driven development (Strands), role-based multi-agent systems (CrewAI), and foundation abstractions (LangChain).

## Further Reading

- [LangGraph](langgraph/ReadMe.md) - Graph-based agent orchestration
- [Strands](strands/ReadMe.md) - Model-driven agent development
- [CrewAI](crew-ai/ReadMe.md) - Role-based multi-agent orchestration
- [LangChain](langchain/ReadMe.md) - Foundation library for LLM applications
