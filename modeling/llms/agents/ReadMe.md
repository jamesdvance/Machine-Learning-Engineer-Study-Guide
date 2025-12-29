# LLM Agents

## Summary

LLM agents are autonomous systems that use large language models as their reasoning core, combining language understanding with tool use, memory, and planning to accomplish complex tasks. Unlike basic LLM applications that respond to single prompts, agents operate in loops, taking actions, observing results, and adapting their behavior. They represent the frontier of practical LLM applications, enabling automation of multi-step workflows that previously required human intervention.

Key points to remember:

- Agents combine LLMs with tools, memory, and planning for autonomous task completion
- The core agent loop: observe, think, act, observe (repeat until done)
- Tool use extends LLM capabilities beyond text generation
- Memory systems enable learning and context persistence across interactions
- Planning enables decomposition and execution of complex multi-step tasks
- Design patterns (ReAct, Plan-and-Execute, Reflexion) provide proven architectures
- Production agents require robust error handling, observability, and security

## What Makes an Agent

### Components of an LLM Agent

```
+------------------+
|   User/System    |
|     Request      |
+--------+---------+
         |
         v
+--------+---------+
|    LLM Core      |  <-- Reasoning and decision-making
|   (Foundation)   |
+--------+---------+
         |
    +----+----+----+----+
    |         |         |
    v         v         v
+-------+ +-------+ +-------+
| Tools | |Memory | |Planning|
+-------+ +-------+ +-------+
    |         |         |
    v         v         v
+------------------+
|   Environment    |
|  (APIs, Files,   |
|   Databases)     |
+------------------+
```

### Agent vs Traditional LLM Application

| Aspect | Traditional LLM | LLM Agent |
|--------|----------------|-----------|
| Interaction | Single prompt-response | Multi-turn loop |
| State | Stateless (context window only) | Persistent memory |
| Actions | Text generation only | Tool execution |
| Complexity | Simple tasks | Multi-step workflows |
| Autonomy | Human-directed | Goal-directed |
| Adaptation | None | Learns from outcomes |

## The Agent Loop

### Basic Implementation

```python
class Agent:
    def __init__(self, llm, tools, memory, planner=None):
        self.llm = llm
        self.tools = {t.name: t for t in tools}
        self.memory = memory
        self.planner = planner

    def run(self, task: str, max_iterations: int = 20):
        """Execute the agent loop until task completion."""
        # Initialize
        self.memory.add_task(task)
        plan = self.planner.create(task) if self.planner else None

        for iteration in range(max_iterations):
            # Observe: Gather context
            context = self._build_context(task, plan)

            # Think: Decide next action
            thought, action = self._reason(context)

            # Check for completion
            if action.type == "finish":
                return action.result

            # Act: Execute action
            observation = self._execute(action)

            # Update memory
            self.memory.add_step(thought, action, observation)

            # Optionally update plan
            if self.planner and not observation.success:
                plan = self.planner.replan(task, self.memory.get_history())

        return "Max iterations reached without completion"

    def _build_context(self, task, plan):
        """Assemble context from memory and environment."""
        return {
            "task": task,
            "plan": plan,
            "history": self.memory.get_recent(k=10),
            "relevant_memories": self.memory.retrieve(task),
            "available_tools": list(self.tools.keys())
        }

    def _reason(self, context):
        """Use LLM to decide next action."""
        prompt = self._format_prompt(context)
        response = self.llm.generate(prompt)
        return self._parse_response(response)

    def _execute(self, action):
        """Execute the chosen action."""
        if action.tool not in self.tools:
            return Observation(success=False, error=f"Unknown tool: {action.tool}")

        try:
            result = self.tools[action.tool].execute(**action.params)
            return Observation(success=True, result=result)
        except Exception as e:
            return Observation(success=False, error=str(e))
```

## Agent Architectures

### ReAct (Reasoning + Acting)

The most common pattern, interleaving explicit reasoning with actions:

```python
REACT_PROMPT = """You are an agent that thinks step-by-step.

Format:
Thought: [Your reasoning about what to do next]
Action: [tool_name(param1="value1", param2="value2")]
Observation: [Result from the tool - provided by system]
... (repeat until done)
Thought: I have completed the task
Action: finish(result="final answer")

Task: {task}
{history}
"""

class ReActAgent(Agent):
    def _format_prompt(self, context):
        history = "\n".join([
            f"Thought: {s.thought}\nAction: {s.action}\nObservation: {s.observation}"
            for s in context["history"]
        ])
        return REACT_PROMPT.format(task=context["task"], history=history)
```

### Plan-and-Execute

Separate planning from execution for complex tasks:

```python
class PlanExecuteAgent(Agent):
    def run(self, task: str, max_iterations: int = 20):
        # Phase 1: Create comprehensive plan
        plan = self.planner.create_detailed(task)

        # Phase 2: Execute each step
        for step in plan.steps:
            result = self._execute_step(step)

            if not result.success:
                # Replan from current state
                remaining_plan = self.planner.replan(
                    task,
                    completed=plan.completed_steps,
                    failed_step=step,
                    error=result.error
                )
                plan = remaining_plan

        return self._synthesize_results(plan)
```

### Multi-Agent Systems

Coordinate multiple specialized agents:

```python
class MultiAgentOrchestrator:
    def __init__(self, agents: Dict[str, Agent], router_llm):
        self.agents = agents
        self.router = router_llm

    def run(self, task: str):
        # Route to appropriate agent(s)
        assignments = self._route_task(task)

        results = {}
        for agent_name, subtask in assignments.items():
            results[agent_name] = self.agents[agent_name].run(subtask)

        # Synthesize results
        return self._synthesize(task, results)

    def _route_task(self, task):
        prompt = f"""Given this task, assign to specialists:

Task: {task}

Available specialists:
{self._format_agents()}

Return assignments as: specialist_name: subtask"""

        response = self.router.generate(prompt)
        return self._parse_assignments(response)
```

## Integration Points

### Tools

Tools extend agent capabilities beyond text:

```python
# Core tools for most agents
essential_tools = [
    WebSearchTool(),      # Information retrieval
    CodeExecutorTool(),   # Computation
    FileSystemTool(),     # Persistence
    APICallTool(),        # External services
]

# Domain-specific tools
domain_tools = [
    DatabaseQueryTool(),  # Data access
    ImageAnalysisTool(),  # Vision
    EmailTool(),          # Communication
]
```

See [Tool Use](tool-use/ReadMe.md) for detailed implementation patterns.

### Memory

Memory enables context and learning:

```python
class AgentMemory:
    def __init__(self):
        self.short_term = ConversationBuffer()   # Current session
        self.working = WorkingScratchpad()       # Current task
        self.long_term = VectorStore()           # Persistent knowledge
        self.episodic = EpisodeStore()           # Past experiences

    def get_context(self, query: str) -> str:
        """Build context from all memory types."""
        return {
            "recent": self.short_term.get_last(5),
            "working": self.working.get_state(),
            "relevant": self.long_term.search(query, k=5),
            "similar_tasks": self.episodic.find_similar(query, k=3)
        }
```

See [Memory Systems](memory-systems/ReadMe.md) for comprehensive coverage.

### Planning

Planning enables complex task decomposition:

```python
class AgentPlanner:
    def create(self, task: str) -> Plan:
        """Decompose task into executable steps."""
        subtasks = self.decompose(task)
        dependencies = self.identify_dependencies(subtasks)
        return Plan(subtasks, dependencies)

    def replan(self, task: str, history: List, error: str) -> Plan:
        """Create new plan given execution history and failure."""
        return self.create_from_context(task, history, error)
```

See [Planning](planning/ReadMe.md) for planning strategies.

### Design Patterns

Proven architectures for different scenarios:

| Pattern | Best For | Trade-off |
|---------|----------|-----------|
| ReAct | Transparency, debugging | More verbose |
| Plan-Execute | Complex multi-step tasks | Higher latency |
| Reflexion | Quality-critical outputs | More iterations |
| Hierarchical | Large-scale tasks | More complexity |

See [Design Patterns](design-patterns/ReadMe.md) for detailed patterns.

## Production Considerations

### Error Handling

```python
class RobustAgent(Agent):
    def __init__(self, *args, max_retries=3, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_retries = max_retries

    def _execute_with_retry(self, action):
        for attempt in range(self.max_retries):
            result = self._execute(action)
            if result.success:
                return result

            # Log and maybe adjust
            self.logger.warning(f"Attempt {attempt+1} failed: {result.error}")

            # Try to recover
            if self._can_recover(result.error):
                action = self._adjust_action(action, result.error)
            else:
                break

        return result

    def _can_recover(self, error):
        """Determine if error is recoverable."""
        recoverable_errors = ["timeout", "rate_limit", "temporary"]
        return any(e in str(error).lower() for e in recoverable_errors)
```

### Observability

```python
class ObservableAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = AgentMetrics()
        self.tracer = AgentTracer()

    def run(self, task: str, **kwargs):
        with self.tracer.trace(task) as trace:
            self.metrics.task_started()

            try:
                result = super().run(task, **kwargs)
                self.metrics.task_completed(success=True)
                return result
            except Exception as e:
                self.metrics.task_completed(success=False)
                trace.record_error(e)
                raise
```

### Security

```python
class SecureAgent(Agent):
    def __init__(self, *args, permissions: set, **kwargs):
        super().__init__(*args, **kwargs)
        self.permissions = permissions

    def _execute(self, action):
        # Check permissions
        required = self.tools[action.tool].required_permissions
        if not required.issubset(self.permissions):
            return Observation(
                success=False,
                error=f"Permission denied for {action.tool}"
            )

        # Sandbox execution
        with self._sandbox():
            return super()._execute(action)
```

## Evaluation

### Agent Benchmarks

| Benchmark | Focus | Metrics |
|-----------|-------|---------|
| WebArena | Web browsing | Task success rate |
| SWE-bench | Code repair | % tests passed |
| AgentBench | General agents | Multi-metric |
| GAIA | Real-world tasks | Accuracy |

### Evaluation Dimensions

```python
def evaluate_agent(agent, test_cases):
    metrics = {
        "task_success_rate": 0,
        "avg_steps": 0,
        "avg_tool_calls": 0,
        "error_rate": 0,
        "avg_latency": 0
    }

    for case in test_cases:
        result = agent.run(case.task)

        metrics["task_success_rate"] += (result == case.expected)
        metrics["avg_steps"] += agent.steps_taken
        metrics["avg_tool_calls"] += agent.tool_calls
        metrics["error_rate"] += agent.errors_encountered
        metrics["avg_latency"] += agent.total_time

    # Normalize
    n = len(test_cases)
    return {k: v/n for k, v in metrics.items()}
```

## When to Use Agents

### Good Fit

- Multi-step tasks requiring tool use
- Tasks needing real-time information
- Workflows with conditional logic
- Tasks benefiting from memory/context
- Automation of complex processes

### Poor Fit

- Simple Q&A (use basic LLM)
- Latency-critical applications
- Tasks requiring perfect reliability
- Highly regulated environments
- Very narrow, well-defined tasks (use traditional software)

## Further Reading

For detailed information on agent components, see:

- [Design Patterns](design-patterns/ReadMe.md) - Agent architectures (ReAct, Plan-Execute, Reflexion)
- [Memory Systems](memory-systems/ReadMe.md) - Short-term, long-term, and working memory
- [Planning](planning/ReadMe.md) - Task decomposition and execution strategies
- [Tool Use](tool-use/ReadMe.md) - Tool definition, execution, and security
