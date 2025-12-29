# Planning for LLM Agents

## Summary

Planning enables LLM agents to decompose complex tasks into manageable steps and execute them systematically. While LLMs can reason about tasks, they struggle with long-horizon planning without explicit structure. Planning architectures address this by separating high-level strategy from step-by-step execution, enabling agents to handle tasks that require multiple coordinated actions, adapt when steps fail, and maintain coherent progress toward goals.

Key points to remember:

- Planning separates "what to do" from "how to do it"
- Task decomposition breaks complex goals into actionable subtasks
- Plans can be static (generated upfront) or dynamic (revised during execution)
- Replanning handles failures and changing conditions
- Hierarchical planning manages complexity through abstraction levels
- World models enable agents to simulate outcomes before acting
- Planning quality directly impacts agent reliability on multi-step tasks

## Why Planning Matters

### The Planning Problem

LLMs face fundamental challenges with complex tasks:

```
Without Planning:
User: "Set up a new Python project with testing, CI, and deployment"

Agent attempts everything at once:
- Creates files in wrong order
- Misses dependencies between steps
- Fails to recover from errors
- Loses track of progress

With Planning:
Agent creates structured plan:
1. Initialize project structure
2. Set up virtual environment
3. Create base package
4. Add testing framework
5. Configure CI pipeline
6. Set up deployment scripts
7. Verify everything works

Each step builds on previous, with clear checkpoints
```

### Planning vs Direct Execution

| Aspect | Direct Execution | With Planning |
|--------|------------------|---------------|
| Simple tasks | Efficient | Overhead |
| Complex tasks | Often fails | More reliable |
| Error recovery | Ad-hoc | Systematic |
| Progress tracking | Implicit | Explicit |
| Transparency | Low | High |

## Task Decomposition

### Basic Decomposition

```python
class TaskDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose(self, task, max_subtasks=10):
        """Break a complex task into subtasks."""
        prompt = f"""Break down this task into specific, actionable subtasks.

Task: {task}

Requirements:
- Each subtask should be independently verifiable
- Order subtasks by dependency (do X before Y if Y needs X)
- Be specific enough that each subtask has clear completion criteria
- Limit to {max_subtasks} subtasks maximum

Return as a numbered list."""

        response = self.llm.generate(prompt)
        return self._parse_subtasks(response)

    def _parse_subtasks(self, response):
        """Extract numbered subtasks from response."""
        subtasks = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and line[0].isdigit():
                # Remove number prefix
                content = line.split('.', 1)[-1].strip()
                subtasks.append({
                    'description': content,
                    'status': 'pending',
                    'dependencies': []
                })
        return subtasks
```

### Dependency-Aware Decomposition

```python
class DependencyAwareDecomposer:
    def __init__(self, llm):
        self.llm = llm

    def decompose_with_dependencies(self, task):
        """Decompose task and identify dependencies."""
        prompt = f"""Break down this task and identify dependencies.

Task: {task}

For each subtask, specify:
- ID: A short identifier (e.g., "setup-env")
- Description: What needs to be done
- Depends-on: IDs of subtasks that must complete first (or "none")

Format each subtask as:
ID | Description | Depends-on

Example:
setup-env | Create virtual environment | none
install-deps | Install dependencies | setup-env
run-tests | Run test suite | install-deps"""

        response = self.llm.generate(prompt)
        return self._parse_with_dependencies(response)

    def _parse_with_dependencies(self, response):
        subtasks = {}
        for line in response.strip().split('\n'):
            if '|' in line:
                parts = [p.strip() for p in line.split('|')]
                if len(parts) >= 3:
                    task_id = parts[0]
                    subtasks[task_id] = {
                        'id': task_id,
                        'description': parts[1],
                        'depends_on': [] if parts[2].lower() == 'none'
                                      else [d.strip() for d in parts[2].split(',')],
                        'status': 'pending'
                    }
        return subtasks

    def get_execution_order(self, subtasks):
        """Topological sort for execution order."""
        order = []
        visited = set()
        temp_visited = set()

        def visit(task_id):
            if task_id in temp_visited:
                raise ValueError(f"Circular dependency detected at {task_id}")
            if task_id in visited:
                return

            temp_visited.add(task_id)
            for dep in subtasks[task_id]['depends_on']:
                if dep in subtasks:
                    visit(dep)
            temp_visited.remove(task_id)
            visited.add(task_id)
            order.append(task_id)

        for task_id in subtasks:
            if task_id not in visited:
                visit(task_id)

        return order
```

## Plan Representation

### Simple Linear Plan

```python
@dataclass
class PlanStep:
    id: str
    description: str
    action: str  # Tool or action to execute
    parameters: Dict[str, Any]
    expected_outcome: str
    status: str = 'pending'
    result: Any = None

class LinearPlan:
    def __init__(self, goal: str, steps: List[PlanStep]):
        self.goal = goal
        self.steps = steps
        self.current_step = 0

    def get_next_step(self) -> Optional[PlanStep]:
        """Get next pending step."""
        for step in self.steps[self.current_step:]:
            if step.status == 'pending':
                return step
        return None

    def complete_step(self, step_id: str, result: Any, success: bool):
        """Mark step as complete."""
        for step in self.steps:
            if step.id == step_id:
                step.status = 'completed' if success else 'failed'
                step.result = result
                if success:
                    self.current_step = self.steps.index(step) + 1
                return

    def is_complete(self) -> bool:
        return all(s.status == 'completed' for s in self.steps)

    def get_progress(self) -> Dict:
        completed = sum(1 for s in self.steps if s.status == 'completed')
        return {
            'completed': completed,
            'total': len(self.steps),
            'percentage': completed / len(self.steps) * 100
        }
```

### DAG-Based Plan

```python
class DAGPlan:
    """Plan with parallel execution support."""

    def __init__(self, goal: str):
        self.goal = goal
        self.steps = {}  # id -> PlanStep
        self.dependencies = {}  # id -> set of dependency ids

    def add_step(self, step: PlanStep, depends_on: List[str] = None):
        self.steps[step.id] = step
        self.dependencies[step.id] = set(depends_on or [])

    def get_ready_steps(self) -> List[PlanStep]:
        """Get all steps whose dependencies are satisfied."""
        ready = []
        for step_id, step in self.steps.items():
            if step.status != 'pending':
                continue

            deps_satisfied = all(
                self.steps[dep].status == 'completed'
                for dep in self.dependencies[step_id]
                if dep in self.steps
            )

            if deps_satisfied:
                ready.append(step)

        return ready

    def can_parallelize(self) -> bool:
        """Check if multiple steps can run in parallel."""
        return len(self.get_ready_steps()) > 1
```

### Hierarchical Plan

```python
class HierarchicalPlan:
    """Multi-level plan with abstraction hierarchy."""

    def __init__(self, goal: str):
        self.goal = goal
        self.levels = {}  # level_num -> list of plan nodes
        self.current_level = 0

    def add_level(self, level_num: int, steps: List[PlanStep]):
        """Add a planning level."""
        self.levels[level_num] = steps

    def expand_step(self, step_id: str, substeps: List[PlanStep]):
        """Expand a high-level step into substeps."""
        # Find the step
        for level_num, steps in self.levels.items():
            for i, step in enumerate(steps):
                if step.id == step_id:
                    # Create new level for substeps
                    new_level = level_num + 0.5  # Insert between levels
                    self.levels[new_level] = substeps
                    step.expanded = True
                    step.substep_ids = [s.id for s in substeps]
                    return

    def get_current_steps(self) -> List[PlanStep]:
        """Get steps at the current execution level."""
        levels = sorted(self.levels.keys())
        if self.current_level < len(levels):
            return self.levels[levels[self.current_level]]
        return []
```

## Plan Generation

### LLM-Based Planning

```python
class LLMPlanner:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = tools
        self.tool_descriptions = self._format_tools()

    def _format_tools(self):
        return "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools
        ])

    def generate_plan(self, task, context=None):
        """Generate a plan for the given task."""
        prompt = f"""Create a detailed plan to accomplish this task.

Task: {task}

Available tools:
{self.tool_descriptions}

{"Context: " + context if context else ""}

Create a step-by-step plan where each step specifies:
1. What action to take
2. Which tool to use (if any)
3. What parameters to provide
4. What outcome to expect
5. How to verify success

Format each step as:
Step N: [Description]
- Tool: [tool_name or "none"]
- Parameters: [param1=value1, param2=value2]
- Expected: [expected outcome]
- Verify: [how to verify success]"""

        response = self.llm.generate(prompt)
        return self._parse_plan(response, task)

    def _parse_plan(self, response, goal):
        """Parse LLM response into structured plan."""
        steps = []
        current_step = None

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('Step'):
                if current_step:
                    steps.append(current_step)
                current_step = {
                    'description': line.split(':', 1)[-1].strip(),
                    'tool': None,
                    'parameters': {},
                    'expected': '',
                    'verify': ''
                }
            elif current_step:
                if line.startswith('- Tool:'):
                    current_step['tool'] = line.split(':')[-1].strip()
                elif line.startswith('- Parameters:'):
                    params_str = line.split(':')[-1].strip()
                    current_step['parameters'] = self._parse_params(params_str)
                elif line.startswith('- Expected:'):
                    current_step['expected'] = line.split(':')[-1].strip()
                elif line.startswith('- Verify:'):
                    current_step['verify'] = line.split(':')[-1].strip()

        if current_step:
            steps.append(current_step)

        return LinearPlan(
            goal=goal,
            steps=[
                PlanStep(
                    id=f"step_{i}",
                    description=s['description'],
                    action=s['tool'],
                    parameters=s['parameters'],
                    expected_outcome=s['expected']
                )
                for i, s in enumerate(steps)
            ]
        )
```

### Plan Validation

```python
class PlanValidator:
    def __init__(self, tools):
        self.tools = {t.name: t for t in tools}

    def validate(self, plan):
        """Validate plan for executability."""
        issues = []

        for step in plan.steps:
            # Check tool exists
            if step.action and step.action != 'none':
                if step.action not in self.tools:
                    issues.append({
                        'step': step.id,
                        'issue': f"Unknown tool: {step.action}",
                        'severity': 'error'
                    })
                else:
                    # Check parameters
                    tool = self.tools[step.action]
                    missing = self._check_required_params(tool, step.parameters)
                    if missing:
                        issues.append({
                            'step': step.id,
                            'issue': f"Missing parameters: {missing}",
                            'severity': 'error'
                        })

            # Check for vague descriptions
            if len(step.description) < 10:
                issues.append({
                    'step': step.id,
                    'issue': 'Description too vague',
                    'severity': 'warning'
                })

        return {
            'valid': not any(i['severity'] == 'error' for i in issues),
            'issues': issues
        }

    def _check_required_params(self, tool, provided_params):
        """Check if all required parameters are provided."""
        required = getattr(tool, 'required_params', [])
        missing = [p for p in required if p not in provided_params]
        return missing
```

## Plan Execution

### Basic Executor

```python
class PlanExecutor:
    def __init__(self, tools, llm=None):
        self.tools = {t.name: t for t in tools}
        self.llm = llm
        self.execution_history = []

    def execute(self, plan, on_step_complete=None):
        """Execute a plan step by step."""
        while not plan.is_complete():
            step = plan.get_next_step()
            if not step:
                break

            # Execute the step
            result = self._execute_step(step)

            # Record execution
            self.execution_history.append({
                'step': step,
                'result': result,
                'timestamp': datetime.now()
            })

            # Update plan
            plan.complete_step(step.id, result, result['success'])

            # Callback
            if on_step_complete:
                on_step_complete(step, result)

            # Handle failure
            if not result['success']:
                recovery = self._attempt_recovery(step, result)
                if not recovery['recovered']:
                    return {
                        'success': False,
                        'failed_step': step,
                        'error': result['error']
                    }

        return {
            'success': plan.is_complete(),
            'results': self.execution_history
        }

    def _execute_step(self, step):
        """Execute a single step."""
        try:
            if step.action and step.action in self.tools:
                tool = self.tools[step.action]
                output = tool.execute(**step.parameters)
                return {'success': True, 'output': output}
            elif step.action == 'none' or not step.action:
                # No tool needed, just acknowledge
                return {'success': True, 'output': 'Step completed'}
            else:
                return {'success': False, 'error': f'Unknown tool: {step.action}'}

        except Exception as e:
            return {'success': False, 'error': str(e)}
```

### Parallel Executor

```python
import asyncio
from concurrent.futures import ThreadPoolExecutor

class ParallelPlanExecutor:
    def __init__(self, tools, max_parallel=4):
        self.tools = {t.name: t for t in tools}
        self.max_parallel = max_parallel
        self.executor = ThreadPoolExecutor(max_workers=max_parallel)

    async def execute(self, plan: DAGPlan):
        """Execute a DAG plan with parallelism."""
        results = {}

        while True:
            ready_steps = plan.get_ready_steps()
            if not ready_steps:
                break

            # Execute ready steps in parallel
            tasks = [
                self._execute_step_async(step)
                for step in ready_steps[:self.max_parallel]
            ]

            step_results = await asyncio.gather(*tasks)

            # Update plan with results
            for step, result in zip(ready_steps, step_results):
                plan.steps[step.id].status = 'completed' if result['success'] else 'failed'
                plan.steps[step.id].result = result
                results[step.id] = result

        return results

    async def _execute_step_async(self, step):
        """Execute a step asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            lambda: self._execute_step(step)
        )
```

## Replanning

### Dynamic Replanning

```python
class AdaptivePlanner:
    def __init__(self, llm, tools):
        self.llm = llm
        self.planner = LLMPlanner(llm, tools)
        self.executor = PlanExecutor(tools, llm)

    def execute_with_replanning(self, task, max_replans=3):
        """Execute task with dynamic replanning on failure."""
        plan = self.planner.generate_plan(task)
        execution_history = []
        replan_count = 0

        while replan_count < max_replans:
            result = self.executor.execute(plan)

            if result['success']:
                return {
                    'success': True,
                    'replans': replan_count,
                    'history': execution_history
                }

            # Plan failed, attempt replan
            execution_history.extend(result.get('results', []))
            failed_step = result.get('failed_step')
            error = result.get('error')

            # Generate new plan from current state
            plan = self._replan(task, execution_history, failed_step, error)
            replan_count += 1

        return {
            'success': False,
            'replans': replan_count,
            'error': 'Max replans exceeded'
        }

    def _replan(self, original_task, history, failed_step, error):
        """Generate a new plan given execution history."""
        completed_steps = [h['step'].description for h in history
                         if h['result']['success']]

        prompt = f"""The original task was: {original_task}

These steps were completed successfully:
{chr(10).join(f"- {s}" for s in completed_steps)}

This step failed: {failed_step.description}
Error: {error}

Create a new plan to complete the remaining work, accounting for the failure.
Consider alternative approaches to avoid the same error."""

        return self.planner.generate_plan(prompt)
```

### Continuous Monitoring and Adjustment

```python
class MonitoredExecutor:
    def __init__(self, planner, executor, monitor):
        self.planner = planner
        self.executor = executor
        self.monitor = monitor

    def execute_with_monitoring(self, plan):
        """Execute with continuous condition monitoring."""
        while not plan.is_complete():
            step = plan.get_next_step()

            # Check preconditions
            if not self.monitor.check_preconditions(step):
                # Conditions changed, may need to replan
                plan = self._adapt_plan(plan, step, 'precondition_failed')
                continue

            # Execute step
            result = self.executor._execute_step(step)

            # Check if result matches expectations
            if result['success']:
                if not self.monitor.verify_outcome(step, result):
                    # Unexpected outcome
                    plan = self._adapt_plan(plan, step, 'unexpected_outcome')
                    continue

            plan.complete_step(step.id, result, result['success'])

            # Check if overall conditions still valid
            if not self.monitor.check_global_conditions():
                plan = self._adapt_plan(plan, step, 'conditions_changed')

        return {'success': plan.is_complete()}

    def _adapt_plan(self, current_plan, current_step, reason):
        """Adapt plan based on monitoring feedback."""
        context = {
            'completed_steps': [s for s in current_plan.steps if s.status == 'completed'],
            'current_step': current_step,
            'reason': reason,
            'current_conditions': self.monitor.get_current_state()
        }
        return self.planner.replan(current_plan.goal, context)
```

## World Models for Planning

### Simulated Planning

```python
class WorldModelPlanner:
    """Planner that simulates outcomes before executing."""

    def __init__(self, llm, tools, world_model):
        self.llm = llm
        self.tools = tools
        self.world_model = world_model

    def plan_with_simulation(self, task, initial_state):
        """Generate plan by simulating different approaches."""
        # Generate candidate plans
        candidates = self._generate_candidates(task, n=3)

        # Simulate each plan
        simulations = []
        for plan in candidates:
            sim_result = self._simulate(plan, initial_state)
            simulations.append({
                'plan': plan,
                'predicted_success': sim_result['success'],
                'predicted_outcome': sim_result['final_state'],
                'confidence': sim_result['confidence']
            })

        # Select best plan
        best = max(simulations, key=lambda x: x['confidence'] * x['predicted_success'])
        return best['plan']

    def _simulate(self, plan, initial_state):
        """Simulate plan execution using world model."""
        state = initial_state.copy()
        confidence = 1.0

        for step in plan.steps:
            # Predict outcome of this step
            prediction = self.world_model.predict(state, step.action, step.parameters)

            # Update state
            state = prediction['next_state']
            confidence *= prediction['confidence']

            # Check for predicted failure
            if prediction['failure_probability'] > 0.5:
                return {
                    'success': False,
                    'final_state': state,
                    'confidence': confidence,
                    'failed_at': step.id
                }

        return {
            'success': True,
            'final_state': state,
            'confidence': confidence
        }

    def _generate_candidates(self, task, n):
        """Generate multiple candidate plans."""
        candidates = []
        for i in range(n):
            prompt = f"""Generate plan #{i+1} for: {task}

Consider a {'different' if i > 0 else 'straightforward'} approach."""

            plan = self.planner.generate_plan(prompt)
            candidates.append(plan)

        return candidates
```

### Learning from Execution

```python
class LearningPlanner:
    """Planner that improves from execution experience."""

    def __init__(self, llm, tools, experience_store):
        self.llm = llm
        self.tools = tools
        self.experience_store = experience_store

    def plan_with_experience(self, task):
        """Generate plan informed by past experience."""
        # Retrieve similar past tasks
        similar = self.experience_store.retrieve_similar(task, k=5)

        # Extract lessons
        lessons = self._extract_lessons(similar)

        # Generate plan with lessons
        prompt = f"""Plan this task: {task}

Lessons from similar past tasks:
{lessons}

Apply these lessons to create a robust plan."""

        return self.planner.generate_plan(prompt)

    def record_execution(self, task, plan, result):
        """Record execution for future learning."""
        experience = {
            'task': task,
            'plan': plan,
            'success': result['success'],
            'steps_completed': len([r for r in result.get('results', [])
                                   if r['result']['success']]),
            'failure_reason': result.get('error'),
            'timestamp': datetime.now()
        }

        self.experience_store.add(experience)

    def _extract_lessons(self, similar_experiences):
        """Extract actionable lessons from past experiences."""
        successes = [e for e in similar_experiences if e['success']]
        failures = [e for e in similar_experiences if not e['success']]

        lessons = []
        if successes:
            lessons.append("Successful approaches:")
            for exp in successes[:2]:
                lessons.append(f"  - {exp['plan'].goal}: worked well")

        if failures:
            lessons.append("Approaches to avoid:")
            for exp in failures[:2]:
                lessons.append(f"  - {exp['failure_reason']}")

        return "\n".join(lessons)
```

## Planning Strategies

### Strategy Comparison

| Strategy | Best For | Trade-offs |
|----------|----------|------------|
| Linear | Simple sequential tasks | No parallelism, brittle |
| DAG-based | Tasks with parallel opportunities | More complex, better throughput |
| Hierarchical | Large complex tasks | More overhead, better organization |
| Iterative | Exploratory tasks | Slower, more adaptive |
| Simulation-based | High-stakes decisions | Requires world model, more compute |

### Choosing a Strategy

```python
def recommend_planning_strategy(task_analysis):
    """Recommend planning strategy based on task characteristics."""

    if task_analysis['step_count'] <= 3:
        return 'linear'  # Simple enough for linear

    if task_analysis['has_parallel_opportunities']:
        return 'dag'  # Exploit parallelism

    if task_analysis['complexity'] == 'high':
        if task_analysis['can_decompose_hierarchically']:
            return 'hierarchical'
        else:
            return 'iterative'  # Break down as we go

    if task_analysis['high_failure_cost']:
        return 'simulation'  # Simulate before acting

    return 'linear'  # Default fallback
```

## Key Takeaways

1. **Decompose before executing**: Break complex tasks into manageable subtasks with clear dependencies.

2. **Plan representation matters**: Choose linear, DAG, or hierarchical based on task structure.

3. **Validate plans**: Check for unknown tools, missing parameters, and logical issues before execution.

4. **Enable replanning**: Expect failures and design for adaptation.

5. **Exploit parallelism**: Use DAG plans to execute independent steps concurrently.

6. **Learn from experience**: Record execution outcomes to improve future planning.

7. **Simulate when stakes are high**: Use world models to predict outcomes before committing to actions.
