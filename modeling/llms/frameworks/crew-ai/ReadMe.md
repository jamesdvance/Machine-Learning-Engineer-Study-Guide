# CrewAI

## Summary

CrewAI is a Python framework for building multi-agent AI systems where specialized agents collaborate as a team to accomplish complex tasks. Built independently from LangChain, it provides role-based agent orchestration with a focus on enterprise workflows and production deployment. The framework models agent collaboration similarly to corporate team structures, where agents with defined roles, goals, and expertise work together under crew coordination.

Key points to remember:

- Agents are defined with roles, goals, and backstories that shape their behavior and decision-making
- Tasks are discrete work units with expected outputs that agents complete sequentially or in parallel
- Crews orchestrate agent teams using sequential or hierarchical process modes
- Flows provide event-driven workflow orchestration for complex conditional logic beyond crew capabilities
- Memory systems (short-term, long-term, entity) enable agents to learn and improve over time
- YAML-based configuration separates agent and task definitions from code logic
- Supports 40+ built-in tools plus custom tool creation via decorators or class inheritance
- Model-agnostic with support for OpenAI, Anthropic, Google, AWS Bedrock, and local models

## Installation and Setup

CrewAI requires Python 3.10 or higher:

```bash
pip install crewai
```

For the full toolkit including tools:

```bash
pip install 'crewai[tools]'
```

The CLI provides project scaffolding:

```bash
crewai create crew my-project
cd my-project
crewai install
```

This generates a project structure with configuration files, crew definition, and entry point:

```
my-project/
  config/
    agents.yaml
    tasks.yaml
  src/
    my_project/
      crew.py
      main.py
```

## Core Concepts

### Agents

Agents are autonomous units that perform tasks based on their defined characteristics. Three required parameters shape agent behavior:

```python
from crewai import Agent

researcher = Agent(
    role="Senior Research Analyst",
    goal="Discover and analyze emerging market trends",
    backstory="""You are a veteran analyst with 15 years of experience
    in technology markets. You excel at synthesizing complex data into
    actionable insights."""
)
```

The role defines what the agent does, the goal guides its decision-making priorities, and the backstory provides context that influences how it approaches problems.

Key optional parameters:

| Parameter | Purpose | Default |
|-----------|---------|---------|
| llm | Language model for the agent | GPT-4 |
| tools | List of tools the agent can use | Empty |
| max_iter | Maximum reasoning iterations | 20 |
| memory | Enable conversation history | False |
| allow_delegation | Permit delegating to other agents | False |
| verbose | Enable detailed execution logs | False |

### Tasks

Tasks define specific assignments with clear objectives:

```python
from crewai import Task

research_task = Task(
    description="""Analyze the current state of AI agent frameworks.
    Focus on adoption rates, key differentiators, and enterprise use cases.""",
    expected_output="""A detailed report covering:
    - Top 5 frameworks by adoption
    - Key technical differentiators
    - Enterprise deployment patterns""",
    agent=researcher
)
```

The expected_output parameter is required and tells the agent what successful completion looks like. Tasks support several output formats:

```python
from pydantic import BaseModel

class ResearchReport(BaseModel):
    title: str
    key_findings: list[str]
    recommendations: list[str]

task = Task(
    description="...",
    expected_output="A structured research report",
    output_pydantic=ResearchReport,  # Validates output against schema
    agent=researcher
)
```

Tasks can depend on other tasks through the context parameter:

```python
writing_task = Task(
    description="Write an executive summary based on the research",
    expected_output="A one-page executive summary",
    agent=writer,
    context=[research_task]  # Receives output from research_task
)
```

### Crews

Crews coordinate agent teams to complete workflows:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, writing_task],
    process=Process.sequential,  # Tasks execute in order
    verbose=True
)

result = crew.kickoff()
```

Two process modes are available:

Sequential process executes tasks in order, with each task completing before the next begins. This ensures predictable flow where later tasks can depend on earlier outputs.

Hierarchical process introduces a manager agent that delegates work to team members:

```python
from crewai import Crew, Process

crew = Crew(
    agents=[researcher, writer, analyst],
    tasks=[complex_task],
    process=Process.hierarchical,
    manager_llm="gpt-4o"  # Or manager_agent for custom manager
)
```

The manager decides which agents handle which parts of tasks, validates outputs, and coordinates the overall workflow.

## YAML Configuration

Production crews typically separate configuration from code:

```yaml
# config/agents.yaml
researcher:
  role: Senior Research Analyst
  goal: Discover and analyze emerging market trends
  backstory: >
    You are a veteran analyst with deep expertise in technology markets.
    You excel at finding non-obvious insights in complex data.

writer:
  role: Technical Writer
  goal: Create clear, engaging content from technical research
  backstory: >
    You translate complex technical concepts into accessible prose.
    Your writing is known for clarity and precision.
```

```yaml
# config/tasks.yaml
research_task:
  description: >
    Analyze the current state of {topic}.
    Focus on key trends and emerging patterns.
  expected_output: >
    A detailed analysis covering major developments,
    key players, and future outlook.
  agent: researcher

writing_task:
  description: >
    Create an executive summary from the research findings.
  expected_output: >
    A one-page summary suitable for senior leadership.
  agent: writer
  context:
    - research_task
```

The crew loads these configurations using decorators:

```python
from crewai import Agent, Task, Crew, CrewBase

@CrewBase
class ResearchCrew:
    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool()],
            verbose=True
        )

    @agent
    def writer(self) -> Agent:
        return Agent(
            config=self.agents_config['writer'],
            verbose=True
        )

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config['research_task'])

    @task
    def writing_task(self) -> Task:
        return Task(config=self.tasks_config['writing_task'])

    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential
        )
```

## Tools

CrewAI provides 40+ built-in tools across categories:

- Search and research: SerperDevTool, EXASearchTool, WikipediaTools
- File operations: FileReadTool, DirectoryReadTool, PDFSearchTool
- Web scraping: ScrapeWebsiteTool, BrowserbaseLoadTool
- Code execution: CodeInterpreterTool
- Databases: PGSearchTool, MySQLTool

Custom tools use the decorator pattern:

```python
from crewai import tool

@tool("Search Customer Database")
def search_customers(query: str, limit: int = 10) -> list:
    """Search the customer database for records matching the query.

    Args:
        query: Search terms to match against customer records
        limit: Maximum number of results to return

    Returns:
        List of matching customer records
    """
    return db.search(query, limit=limit)
```

The docstring becomes the tool description that helps the LLM understand when to use it. Clear descriptions are essential for effective tool selection.

For more control, subclass BaseTool:

```python
from crewai import BaseTool
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    query: str = Field(description="Search terms")
    filters: dict = Field(default={}, description="Optional filters")

class AdvancedSearchTool(BaseTool):
    name: str = "Advanced Search"
    description: str = "Search with advanced filtering capabilities"
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, filters: dict = {}) -> str:
        results = search_engine.query(query, **filters)
        return format_results(results)
```

Assign tools to agents at initialization:

```python
analyst = Agent(
    role="Data Analyst",
    goal="Extract insights from data sources",
    tools=[search_customers, AdvancedSearchTool()],
    backstory="..."
)
```

## Memory Systems

CrewAI provides multiple memory types that enable agents to learn and improve:

Short-term memory stores recent interactions using RAG, allowing agents to maintain context during the current execution. This helps with multi-step reasoning where earlier observations inform later decisions.

Long-term memory persists insights across executions using SQLite. Agents build knowledge over time, improving their problem-solving based on past experiences.

Entity memory tracks information about people, organizations, and concepts. This enables relationship mapping and deeper contextual understanding.

Enable memory with a single parameter:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True  # Activates all memory types
)
```

Memory files are stored in platform-specific directories:

- macOS: ~/Library/Application Support/CrewAI/
- Linux: ~/.local/share/CrewAI/
- Windows: C:\Users\{username}\AppData\Local\CrewAI\

Override with the CREWAI_STORAGE_DIR environment variable.

Memory uses embeddings for retrieval, defaulting to OpenAI. Configure alternative providers:

```python
crew = Crew(
    agents=[...],
    tasks=[...],
    memory=True,
    embedder={
        "provider": "ollama",
        "config": {
            "model": "nomic-embed-text"
        }
    }
)
```

## Flows

Flows provide event-driven workflow orchestration for complex scenarios that exceed crew capabilities:

```python
from crewai.flow.flow import Flow, listen, start

class ResearchFlow(Flow):
    @start()
    def gather_requirements(self):
        return self.state.requirements

    @listen(gather_requirements)
    def conduct_research(self, requirements):
        # Research based on requirements
        research_crew = ResearchCrew()
        return research_crew.crew().kickoff(inputs={"topic": requirements})

    @listen(conduct_research)
    def review_and_approve(self, research):
        # Conditional routing based on quality
        if research.quality_score > 0.8:
            return "approved"
        return "revision_needed"
```

Flows excel at:

- Conditional branching based on intermediate results
- Human-in-the-loop decision points
- Combining code-based logic with crew execution
- State persistence across workflow restarts

Use the router decorator for conditional paths:

```python
from crewai.flow.flow import router

@router(review_and_approve)
def route_decision(self):
    if self.state.approval_status == "approved":
        return "publish"
    return "revise"

@listen("publish")
def publish_research(self):
    # Publish approved research
    pass

@listen("revise")
def request_revision(self):
    # Send back for revision
    pass
```

State management in flows supports both unstructured and structured approaches:

```python
from pydantic import BaseModel

class ResearchState(BaseModel):
    topic: str = ""
    research_complete: bool = False
    approval_status: str = "pending"

class TypedResearchFlow(Flow[ResearchState]):
    @start()
    def initialize(self):
        self.state.topic = "AI Agents"
```

## Model Configuration

CrewAI supports multiple LLM providers:

```python
from crewai import Agent, LLM

# OpenAI
openai_llm = LLM(model="gpt-4o", temperature=0.7)

# Anthropic
anthropic_llm = LLM(
    model="claude-sonnet-4-20250514",
    api_key="your-api-key"
)

# AWS Bedrock
bedrock_llm = LLM(
    model="bedrock/anthropic.claude-sonnet-4-20250514-v1:0",
    region_name="us-east-1"
)

# Local via Ollama
ollama_llm = LLM(
    model="ollama/llama3",
    base_url="http://localhost:11434"
)

agent = Agent(
    role="...",
    goal="...",
    backstory="...",
    llm=anthropic_llm
)
```

Set a default model via environment variable:

```bash
export OPENAI_MODEL_NAME="gpt-4o"
```

## Execution Patterns

### Synchronous Execution

```python
result = crew.kickoff(inputs={"topic": "AI Trends"})
print(result.raw)  # Raw text output
print(result.json_dict)  # Structured output if configured
print(result.token_usage)  # Token consumption metrics
```

### Asynchronous Execution

```python
import asyncio

async def run_crews():
    crew1 = ResearchCrew().crew()
    crew2 = AnalysisCrew().crew()

    results = await asyncio.gather(
        crew1.akickoff(inputs={"topic": "AI"}),
        crew2.akickoff(inputs={"data": "market_data.csv"})
    )
    return results
```

### Streaming

```python
for event in crew.kickoff(inputs={"topic": "AI"}, stream=True):
    if event.type == "agent_thinking":
        print(f"[{event.agent}] {event.content}")
    elif event.type == "task_complete":
        print(f"Task completed: {event.task}")
```

## Comparison with Other Frameworks

| Aspect | CrewAI | LangGraph | Strands |
|--------|--------|-----------|---------|
| Architecture | Role-based teams | Graph workflows | Model-driven |
| Orchestration | Crews with processes | Explicit edges | Implicit through LLM |
| Configuration | YAML + decorators | Code-based | Code-based |
| Multi-agent | Native with delegation | Subgraphs | A2A protocol |
| Memory | Built-in types | External stores | Sessions |
| Best for | Team-based workflows | Complex control flow | Rapid prototyping |

Choose CrewAI when:

- Building workflows that mirror team structures
- Agents need distinct roles and expertise areas
- Delegation between agents is important
- YAML configuration simplifies management
- Enterprise deployment is the target

Consider alternatives when:

- You need fine-grained control over execution paths (LangGraph)
- Minimal boilerplate is priority (Strands)
- Workflows are highly dynamic rather than role-based

## Enterprise Integrations

CrewAI provides triggers for automating crew execution from external events:

- Email platforms: Gmail, Outlook
- File storage: Google Drive, OneDrive
- Communication: Microsoft Teams
- CRM: HubSpot
- Webhooks for custom integrations

```python
# Crews can be triggered by external events
# Configure via CrewAI Enterprise dashboard or API
trigger_config = {
    "source": "gmail",
    "filter": "subject:urgent",
    "crew": "support_crew"
}
```

The platform also supports calling external AI systems:

```python
from crewai_tools import BedrockAgentTool

# Call Amazon Bedrock agents from within crews
bedrock_tool = BedrockAgentTool(
    agent_id="your-bedrock-agent-id",
    agent_alias_id="your-alias-id"
)

agent = Agent(
    role="Coordinator",
    tools=[bedrock_tool],
    ...
)
```

## Production Considerations

### Observability

CrewAI integrates with multiple observability platforms:

```python
import os

# Langfuse integration
os.environ["LANGFUSE_PUBLIC_KEY"] = "..."
os.environ["LANGFUSE_SECRET_KEY"] = "..."

# Or OpenLIT
os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = "..."
```

### Error Handling

```python
from crewai.exceptions import CrewExecutionError

try:
    result = crew.kickoff(inputs=inputs)
except CrewExecutionError as e:
    logger.error(f"Crew execution failed: {e}")
    # Handle gracefully
```

### Cost Management

Token usage is tracked automatically:

```python
result = crew.kickoff()
print(f"Total tokens: {result.token_usage.total_tokens}")
print(f"Prompt tokens: {result.token_usage.prompt_tokens}")
print(f"Completion tokens: {result.token_usage.completion_tokens}")
```

Set iteration limits to prevent runaway execution:

```python
agent = Agent(
    role="...",
    goal="...",
    backstory="...",
    max_iter=10  # Stop after 10 reasoning iterations
)
```

## Further Reading

- CrewAI Documentation: https://docs.crewai.com
- GitHub Repository: https://github.com/crewAIInc/crewAI
- CrewAI Tools: https://github.com/crewAIInc/crewAI-tools
