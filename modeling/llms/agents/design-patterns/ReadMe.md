# LLM Agent Design Patterns

## Summary

Agent design patterns are reusable architectural approaches for building LLM-powered autonomous systems. These patterns address common challenges like task decomposition, error recovery, and multi-step reasoning. Understanding these patterns helps engineers design robust agents that can handle complex tasks while maintaining reliability and interpretability.

Key points to remember:

- ReAct (Reasoning + Acting) interleaves thinking and action for transparent decision-making
- Reflexion adds self-critique loops to improve outputs iteratively
- Plan-and-Execute separates planning from execution for complex multi-step tasks
- Multi-agent systems distribute work across specialized agents
- Hierarchical agents use manager-worker structures for task delegation
- Error recovery patterns like retry, fallback, and human-in-the-loop improve reliability
- Choose patterns based on task complexity, latency requirements, and reliability needs

## The Core Agent Loop

All agent patterns share a fundamental loop:

```
Observe -> Think -> Act -> Observe -> ...
```

```python
class BaseAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.memory = []

    def run(self, task):
        observation = task
        while not self.is_complete(observation):
            # Think: decide what to do
            thought, action = self.think(observation)

            # Act: execute the action
            result = self.act(action)

            # Update observation
            observation = result
            self.memory.append((thought, action, result))

        return self.format_output()
```

## ReAct Pattern

### Overview

ReAct (Reasoning and Acting) interleaves chain-of-thought reasoning with action execution. The model explicitly states its reasoning before each action, making decisions transparent and debuggable.

### Structure

```
Thought: I need to find the current weather in Tokyo
Action: search_weather(location="Tokyo")
Observation: Temperature: 22C, Sunny
Thought: Now I have the weather, I should format the response
Action: respond("The weather in Tokyo is 22C and sunny")
```

### Implementation

```python
REACT_PROMPT = """Answer the question using the following format:

Thought: Think about what to do next
Action: tool_name(param1="value1", param2="value2")
Observation: Result from the tool
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information
Action: finish(answer="final answer")

Available tools: {tools}

Question: {question}
"""

class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    def run(self, question, max_steps=10):
        history = []

        for step in range(max_steps):
            prompt = self._build_prompt(question, history)
            response = self.llm.generate(prompt)

            thought, action = self._parse_response(response)
            history.append(f"Thought: {thought}")
            history.append(f"Action: {action}")

            if action.startswith("finish"):
                return self._extract_answer(action)

            tool_name, params = self._parse_action(action)
            result = self.tools[tool_name].execute(**params)
            history.append(f"Observation: {result}")

        return "Max steps reached without answer"
```

### When to Use

- Tasks requiring transparent reasoning
- Debugging and interpretability are important
- Moderate complexity tasks (3-10 steps)
- Interactive applications where users see agent thinking

## Plan-and-Execute Pattern

### Overview

Plan-and-Execute separates high-level planning from step-by-step execution. A planner creates a comprehensive plan, then an executor carries out each step. This works well for complex, multi-step tasks.

### Structure

```
Phase 1: Planning
----------
Task: Write a blog post about AI agents
Plan:
1. Research current trends in AI agents
2. Outline the main sections
3. Write the introduction
4. Write each section
5. Add conclusion
6. Review and edit

Phase 2: Execution
----------
Executing step 1: Research current trends...
Result: [research findings]
Executing step 2: Outline main sections...
Result: [outline]
...
```

### Implementation

```python
class PlanAndExecuteAgent:
    def __init__(self, planner_llm, executor_llm, tools):
        self.planner = planner_llm
        self.executor = executor_llm
        self.tools = tools

    def run(self, task):
        # Phase 1: Create plan
        plan = self._create_plan(task)

        # Phase 2: Execute each step
        results = []
        for step in plan:
            result = self._execute_step(step, results)
            results.append(result)

            # Optionally replan if step fails
            if not result.success:
                plan = self._replan(task, results, plan)

        return self._synthesize_results(results)

    def _create_plan(self, task):
        prompt = f"""Create a step-by-step plan for: {task}

        Return a numbered list of specific, actionable steps.
        Each step should be self-contained and verifiable."""

        response = self.planner.generate(prompt)
        return self._parse_plan(response)

    def _execute_step(self, step, previous_results):
        context = self._format_context(previous_results)
        prompt = f"""Execute this step: {step}

        Previous results:
        {context}

        Available tools: {self.tools}"""

        return self.executor.generate(prompt)
```

### When to Use

- Complex tasks with many steps
- Tasks benefiting from upfront planning
- When you need progress tracking
- Situations requiring plan modification

## Reflexion Pattern

### Overview

Reflexion adds self-evaluation and improvement loops. After completing a task, the agent critiques its own output and iteratively improves it. This produces higher quality results through self-correction.

### Structure

```
Attempt 1:
  Response: [initial response]
  Reflection: The response lacks specific examples
  Score: 6/10

Attempt 2:
  Response: [improved response with examples]
  Reflection: Better, but could be more concise
  Score: 8/10

Attempt 3:
  Response: [concise response with examples]
  Reflection: Meets all criteria
  Score: 9/10 -> Accept
```

### Implementation

```python
class ReflexionAgent:
    def __init__(self, llm, evaluator_llm=None, max_iterations=3):
        self.llm = llm
        self.evaluator = evaluator_llm or llm
        self.max_iterations = max_iterations

    def run(self, task, quality_threshold=8):
        attempts = []

        for i in range(self.max_iterations):
            # Generate response
            if i == 0:
                response = self._initial_attempt(task)
            else:
                response = self._improved_attempt(task, attempts)

            # Evaluate and reflect
            evaluation = self._evaluate(task, response)
            reflection = self._reflect(task, response, evaluation)

            attempts.append({
                'response': response,
                'evaluation': evaluation,
                'reflection': reflection
            })

            if evaluation['score'] >= quality_threshold:
                return response

        # Return best attempt
        return max(attempts, key=lambda x: x['evaluation']['score'])['response']

    def _reflect(self, task, response, evaluation):
        prompt = f"""Reflect on this response and identify improvements:

        Task: {task}
        Response: {response}
        Evaluation: {evaluation}

        What specific changes would improve this response?"""

        return self.evaluator.generate(prompt)
```

### When to Use

- Quality is more important than speed
- Tasks with clear evaluation criteria
- Writing, coding, or creative tasks
- When initial outputs often need refinement

## Multi-Agent Pattern

### Overview

Multi-agent systems use multiple specialized agents that collaborate to solve problems. Each agent has a specific role or expertise, and they communicate to accomplish complex tasks.

### Architectures

**Debate Pattern**: Agents argue different positions, synthesizing into better answers.

```python
class DebateSystem:
    def __init__(self, agents, moderator):
        self.agents = agents  # Different perspectives
        self.moderator = moderator

    def run(self, question, rounds=3):
        positions = []

        # Initial positions
        for agent in self.agents:
            position = agent.generate_position(question)
            positions.append(position)

        # Debate rounds
        for round in range(rounds):
            new_positions = []
            for i, agent in enumerate(self.agents):
                other_positions = positions[:i] + positions[i+1:]
                updated = agent.respond_to_others(question, other_positions)
                new_positions.append(updated)
            positions = new_positions

        # Synthesize
        return self.moderator.synthesize(question, positions)
```

**Ensemble Pattern**: Multiple agents attempt the same task, best answer selected.

```python
class EnsembleAgents:
    def __init__(self, agents, selector):
        self.agents = agents
        self.selector = selector

    def run(self, task):
        responses = []
        for agent in self.agents:
            response = agent.run(task)
            responses.append(response)

        # Select best or combine
        return self.selector.select_best(task, responses)
```

**Pipeline Pattern**: Agents process sequentially, each adding value.

```python
class PipelineAgents:
    def __init__(self, stages):
        self.stages = stages  # [(name, agent), ...]

    def run(self, input_data):
        current = input_data
        for name, agent in self.stages:
            current = agent.process(current)
        return current

# Example: Research -> Draft -> Edit -> Format
pipeline = PipelineAgents([
    ("researcher", ResearchAgent()),
    ("writer", WritingAgent()),
    ("editor", EditingAgent()),
    ("formatter", FormattingAgent())
])
```

### When to Use

- Complex tasks requiring diverse expertise
- Tasks benefiting from multiple perspectives
- When single-agent approaches are insufficient
- Scalability and parallelization needed

## Hierarchical Agent Pattern

### Overview

Hierarchical agents use a manager-worker structure. A manager agent breaks down tasks and delegates to specialized worker agents, then synthesizes their outputs.

### Implementation

```python
class HierarchicalAgent:
    def __init__(self, manager_llm, worker_agents):
        self.manager = manager_llm
        self.workers = {w.name: w for w in worker_agents}

    def run(self, task):
        # Manager decomposes task
        subtasks = self._decompose(task)

        # Delegate to workers
        results = {}
        for subtask in subtasks:
            worker_name = self._assign_worker(subtask)
            worker = self.workers[worker_name]
            results[subtask.id] = worker.run(subtask)

        # Manager synthesizes
        return self._synthesize(task, results)

    def _decompose(self, task):
        prompt = f"""Break this task into subtasks for specialists:

        Task: {task}

        Available specialists: {list(self.workers.keys())}

        Return a list of subtasks with assigned specialists."""

        response = self.manager.generate(prompt)
        return self._parse_subtasks(response)

    def _synthesize(self, task, results):
        prompt = f"""Synthesize these results into a final answer:

        Original task: {task}

        Results from specialists:
        {self._format_results(results)}"""

        return self.manager.generate(prompt)
```

### When to Use

- Large complex tasks with clear subtask boundaries
- When specialized expertise is needed for different parts
- Parallel execution opportunities
- Managing complexity through divide-and-conquer

## Error Recovery Patterns

### Retry with Backoff

```python
class RetryAgent:
    def __init__(self, agent, max_retries=3, backoff_factor=2):
        self.agent = agent
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

    def run(self, task):
        last_error = None

        for attempt in range(self.max_retries):
            try:
                return self.agent.run(task)
            except Exception as e:
                last_error = e
                wait_time = self.backoff_factor ** attempt
                time.sleep(wait_time)

        raise last_error
```

### Fallback Chain

```python
class FallbackAgent:
    def __init__(self, primary, fallbacks):
        self.primary = primary
        self.fallbacks = fallbacks

    def run(self, task):
        try:
            return self.primary.run(task)
        except Exception:
            for fallback in self.fallbacks:
                try:
                    return fallback.run(task)
                except Exception:
                    continue

        return self._default_response(task)
```

### Human-in-the-Loop

```python
class HumanInLoopAgent:
    def __init__(self, agent, human_interface, confidence_threshold=0.7):
        self.agent = agent
        self.human = human_interface
        self.threshold = confidence_threshold

    def run(self, task):
        result, confidence = self.agent.run_with_confidence(task)

        if confidence < self.threshold:
            # Request human review
            human_decision = self.human.review(task, result, confidence)
            if human_decision.approved:
                return human_decision.modified_result or result
            else:
                return self.human.provide_answer(task)

        return result
```

## Choosing Patterns

### Decision Framework

| Factor | ReAct | Plan-Execute | Reflexion | Multi-Agent |
|--------|-------|--------------|-----------|-------------|
| Task complexity | Medium | High | Medium | High |
| Transparency | High | Medium | Medium | Low |
| Latency | Medium | High | High | Variable |
| Quality focus | Medium | Medium | High | High |
| Parallelizable | No | Partially | No | Yes |

### Pattern Combinations

Patterns can be combined:

```python
# Hierarchical + ReAct workers
class HierarchicalReActAgent:
    def __init__(self, manager, worker_tools):
        self.manager = manager
        self.workers = [ReActAgent(llm, tools) for tools in worker_tools]

# Plan-and-Execute with Reflexion
class ReflexivePlanAgent:
    def __init__(self, planner, executor, evaluator):
        self.planner = planner
        self.executor = ReflexionAgent(executor, evaluator)

    def run(self, task):
        plan = self.planner.create_plan(task)
        for step in plan:
            self.executor.run(step)  # Each step gets reflexion
```

## Best Practices

### Observability

```python
class ObservableAgent:
    def __init__(self, agent, logger):
        self.agent = agent
        self.logger = logger

    def run(self, task):
        self.logger.log_start(task)

        for step in self.agent.steps():
            self.logger.log_step(step)

        result = self.agent.finalize()
        self.logger.log_complete(result)

        return result
```

### Testing Patterns

```python
def test_agent_pattern():
    # Test with mock tools
    mock_tools = [MockTool("search", returns="test result")]
    agent = ReActAgent(mock_llm, mock_tools)

    result = agent.run("test question")

    assert "test result" in result
    assert len(agent.memory) <= 10  # Max steps respected
```

## Key Takeaways

1. **ReAct for transparency**: Interleaved thinking and acting makes debugging easy.

2. **Plan-and-Execute for complexity**: Separate planning from execution for multi-step tasks.

3. **Reflexion for quality**: Self-critique loops improve output quality.

4. **Multi-agent for scale**: Distribute work across specialized agents.

5. **Hierarchical for organization**: Manager-worker structures handle complex delegation.

6. **Error recovery is essential**: Always include retry, fallback, or human-in-the-loop patterns.

7. **Combine patterns**: Real systems often combine multiple patterns for robustness.
