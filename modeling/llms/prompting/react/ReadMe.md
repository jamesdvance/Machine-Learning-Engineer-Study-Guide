# ReAct Prompting

## Summary

ReAct (Reasoning and Acting) is a prompting paradigm that interleaves reasoning traces with actions, enabling LLMs to dynamically interact with external tools and information sources while maintaining explicit reasoning. Unlike pure chain-of-thought which only reasons, or pure acting which only takes actions, ReAct alternates between thinking about what to do and actually doing it. This creates a synergistic loop where reasoning guides actions and action outcomes inform further reasoning. ReAct is foundational for building agentic LLM systems that can browse the web, query databases, use APIs, and solve multi-step problems requiring external information.

Key points to remember:

- Thought-Action-Observation loop: Reason, act, observe, repeat
- Tool integration: Actions invoke external tools (search, calculators, APIs)
- Grounded reasoning: Observations provide factual grounding
- Traceable decisions: Explicit reasoning explains action choices
- Recoverable from errors: Can reason about failures and try alternatives
- Foundation for agents: Core pattern for tool-using AI systems
- Works with few-shot: Demonstrate the pattern with examples

## The ReAct Pattern

### Three-Part Structure

```
Thought: Reasoning about the current situation and what to do next
Action: External tool invocation with specific input
Observation: Result from the tool/environment

Repeat until task is complete.

Example:
Question: What is the elevation of the capital of France?

Thought: I need to find the capital of France first, then find its elevation.
Action: Search[capital of France]
Observation: Paris is the capital and largest city of France.

Thought: Paris is the capital. Now I need to find the elevation of Paris.
Action: Search[elevation of Paris]
Observation: Paris has an average elevation of 35 meters above sea level.

Thought: I have the answer. Paris, the capital of France, has an elevation of 35 meters.
Action: Finish[35 meters]
```

### Why Interleaving Works

```
Pure Reasoning (CoT) Limitation:
- No access to external facts
- Can hallucinate information
- Limited to knowledge in weights

Pure Acting Limitation:
- Random or uninformed actions
- No explanation of choices
- Hard to recover from errors

ReAct Advantage:
- Reasoning guides action selection
- Actions provide grounding
- Observations correct reasoning
- Explicit trace for debugging
```

## Implementation

### Basic ReAct Loop

```python
class ReActAgent:
    def __init__(self, model, tools, max_steps=10):
        self.model = model
        self.tools = tools  # Dict of tool_name -> function
        self.max_steps = max_steps

    def run(self, question):
        """Execute ReAct loop until completion."""
        prompt = self.build_initial_prompt(question)
        trajectory = []

        for step in range(self.max_steps):
            # Generate thought and action
            response = self.model.generate(prompt)

            # Parse response
            thought = self.extract_thought(response)
            action, action_input = self.extract_action(response)

            trajectory.append({
                'thought': thought,
                'action': action,
                'action_input': action_input
            })

            # Check for termination
            if action == "Finish":
                return action_input, trajectory

            # Execute action
            observation = self.execute_action(action, action_input)
            trajectory[-1]['observation'] = observation

            # Update prompt with observation
            prompt += f"\nThought: {thought}"
            prompt += f"\nAction: {action}[{action_input}]"
            prompt += f"\nObservation: {observation}\n"

        return "Max steps reached", trajectory

    def execute_action(self, action, action_input):
        """Invoke the specified tool."""
        if action not in self.tools:
            return f"Unknown action: {action}"

        try:
            return self.tools[action](action_input)
        except Exception as e:
            return f"Error: {str(e)}"
```

### Few-Shot Prompt Template

```python
REACT_PROMPT = """Answer the following questions by thinking step by step and using available actions.

Available actions:
- Search[query]: Search for information on the web
- Lookup[term]: Look up a term in the current document
- Calculate[expression]: Evaluate a mathematical expression
- Finish[answer]: Return the final answer

Example 1:
Question: What is the atomic number of the element discovered by Marie Curie that is used in cancer treatment?

Thought: I need to find which element Marie Curie discovered that is used in cancer treatment.
Action: Search[Marie Curie discovered element cancer treatment]
Observation: Marie Curie discovered polonium and radium. Radium was historically used in cancer treatment through radiation therapy.

Thought: Radium is the element. Now I need its atomic number.
Action: Search[atomic number of radium]
Observation: Radium is a chemical element with the symbol Ra and atomic number 88.

Thought: I have the answer. The atomic number is 88.
Action: Finish[88]

Example 2:
Question: Who was the president of the US when the company that makes the iPhone was founded?

Thought: First I need to find when Apple (the company that makes iPhone) was founded.
Action: Search[when was Apple Inc founded]
Observation: Apple Inc. was founded on April 1, 1976, by Steve Jobs, Steve Wozniak, and Ronald Wayne.

Thought: Apple was founded in 1976. Now I need to find who was the US president in 1976.
Action: Search[US president 1976]
Observation: Gerald Ford was the 38th president of the United States from 1974 to 1977.

Thought: Gerald Ford was president when Apple was founded in 1976.
Action: Finish[Gerald Ford]

Question: {question}

Thought:"""
```

## Tool Definitions

### Search Tool

```python
def create_search_tool(search_api_key):
    """Create a web search tool."""

    def search(query):
        import requests
        response = requests.get(
            "https://api.search.example.com/search",
            params={"q": query, "key": search_api_key}
        )
        results = response.json()

        # Return top result snippet
        if results.get('items'):
            return results['items'][0]['snippet']
        return "No results found."

    return search
```

### Calculator Tool

```python
def calculator(expression):
    """Safely evaluate mathematical expressions."""
    import ast
    import operator

    # Safe operators
    operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
    }

    def eval_node(node):
        if isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            return operators[type(node.op)](left, right)
        else:
            raise ValueError(f"Unsupported: {node}")

    try:
        tree = ast.parse(expression, mode='eval')
        return str(eval_node(tree.body))
    except Exception as e:
        return f"Error: {str(e)}"
```

### Database Lookup Tool

```python
def create_database_tool(db_connection):
    """Create a database query tool."""

    def lookup(query):
        # Simple keyword search in database
        cursor = db_connection.cursor()

        # Only allow SELECT queries for safety
        if not query.strip().upper().startswith('SELECT'):
            return "Only SELECT queries allowed"

        try:
            cursor.execute(query)
            results = cursor.fetchall()[:5]  # Limit results
            return str(results)
        except Exception as e:
            return f"Query error: {str(e)}"

    return lookup
```

## Parsing and Error Handling

### Robust Action Parsing

```python
import re

def parse_react_response(response):
    """Parse thought and action from model response."""

    # Extract thought
    thought_match = re.search(
        r'Thought:\s*(.+?)(?=\nAction:|$)',
        response,
        re.DOTALL
    )
    thought = thought_match.group(1).strip() if thought_match else ""

    # Extract action and input
    action_match = re.search(
        r'Action:\s*(\w+)\[(.+?)\]',
        response
    )

    if action_match:
        action = action_match.group(1)
        action_input = action_match.group(2)
    else:
        # Fallback: try to find action without brackets
        action_match = re.search(r'Action:\s*(\w+)\s+(.+)', response)
        if action_match:
            action = action_match.group(1)
            action_input = action_match.group(2)
        else:
            action = "Unknown"
            action_input = ""

    return thought, action, action_input
```

### Error Recovery

```python
def handle_action_error(agent, error, trajectory):
    """Help agent recover from action errors."""

    recovery_prompt = f"""
The previous action resulted in an error:
{error}

Think about what went wrong and try a different approach.

Thought:"""

    response = agent.model.generate(
        agent.current_prompt + recovery_prompt
    )

    return response
```

## ReAct Variants

### ReAct with Self-Reflection

```python
REFLECT_REACT_PROMPT = """After each observation, reflect on whether you're making progress.

Question: {question}

Thought: Let me start by understanding what I need to find.
Action: Search[{initial_search}]
Observation: {observation}

Reflection: Am I closer to the answer? What should I try next?
Thought:"""
```

### ReAct with Planning

```python
def plan_then_react(model, question, tools):
    """Create a plan first, then execute with ReAct."""

    # Step 1: Create plan
    planning_prompt = f"""Question: {question}

Create a step-by-step plan to answer this question:
1."""

    plan = model.generate(planning_prompt)

    # Step 2: Execute plan with ReAct
    react_prompt = f"""Question: {question}

Plan:
{plan}

Now execute the plan step by step.

Thought: Following step 1 of my plan...
Action:"""

    return react_loop(model, react_prompt, tools)
```

### Multi-Agent ReAct

```python
class MultiAgentReAct:
    """Multiple specialized agents collaborate."""

    def __init__(self, agents):
        self.agents = agents  # {role: agent}

    def run(self, question):
        # Coordinator decides which agent handles each step
        coordinator_prompt = f"""
Question: {question}

Available agents:
- Researcher: Good at finding information
- Analyst: Good at calculations and analysis
- Writer: Good at synthesizing answers

Which agent should handle this? Explain why.
"""
        trajectory = []
        current_context = question

        while True:
            # Coordinator assigns task
            assignment = self.coordinator.generate(
                coordinator_prompt + f"\nContext: {current_context}"
            )
            agent_name = self.extract_agent(assignment)

            # Selected agent takes action
            result = self.agents[agent_name].step(current_context)
            trajectory.append((agent_name, result))

            if result.get('finished'):
                return result['answer'], trajectory

            current_context = result['observation']
```

## Practical Implementation

### With LangChain

```python
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.llms import OpenAI

# Define tools
tools = [
    Tool(
        name="Search",
        func=search_function,
        description="Search the web for information"
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Evaluate mathematical expressions"
    ),
]

# Initialize ReAct agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run
result = agent.run("What is the square root of the year Apple was founded?")
```

### With OpenAI Function Calling

```python
import openai

functions = [
    {
        "name": "search",
        "description": "Search for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform calculation",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression"}
            },
            "required": ["expression"]
        }
    }
]

def react_with_functions(question):
    messages = [{"role": "user", "content": question}]

    while True:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=messages,
            functions=functions,
            function_call="auto"
        )

        msg = response.choices[0].message

        if msg.get("function_call"):
            # Execute function
            func_name = msg.function_call.name
            func_args = json.loads(msg.function_call.arguments)
            result = execute_function(func_name, func_args)

            # Add to messages
            messages.append(msg)
            messages.append({
                "role": "function",
                "name": func_name,
                "content": result
            })
        else:
            # Final answer
            return msg.content
```

## Benchmarks

### ReAct Performance

| Task | CoT Only | Act Only | ReAct |
|------|----------|----------|-------|
| HotpotQA | 29.4% | 25.7% | 35.1% |
| Fever (fact verification) | 56.3% | 58.9% | 64.6% |
| ALFWorld (embodied) | 22% | 45% | 71% |
| WebShop (e-commerce) | 62.4% | 50.7% | 66.6% |

### Why ReAct Outperforms

```
CoT fails when:
- Knowledge is outdated or missing
- Problem requires current information
- Hallucinations lead to wrong answers

Acting alone fails when:
- No reasoning to guide action selection
- Can't recover from wrong paths
- Random exploration is inefficient

ReAct succeeds because:
- Reasoning prevents random actions
- Actions ground reasoning in reality
- Observations correct wrong assumptions
- Trace enables debugging and improvement
```

## Common Pitfalls

### Failure Modes

```
1. Action format errors
   - Model outputs malformed actions
   - Fix: Stricter parsing, better examples

2. Infinite loops
   - Repeating same action
   - Fix: Detect repetition, force alternatives

3. Premature termination
   - Finishes without enough information
   - Fix: Require confidence before finishing

4. Tool misuse
   - Wrong tool for the task
   - Fix: Better tool descriptions, examples

5. Observation overload
   - Too much info from tools
   - Fix: Summarize or truncate observations
```

### Debugging ReAct Agents

```python
def debug_react_run(question, trajectory):
    """Analyze a ReAct trajectory for issues."""

    print(f"Question: {question}\n")

    for i, step in enumerate(trajectory):
        print(f"Step {i+1}:")
        print(f"  Thought: {step['thought'][:100]}...")
        print(f"  Action: {step['action']}[{step['action_input'][:50]}...]")
        print(f"  Observation: {step.get('observation', 'N/A')[:100]}...")
        print()

    # Check for issues
    actions = [s['action'] for s in trajectory]
    if len(actions) != len(set(actions)):
        print("Warning: Repeated actions detected")

    if len(trajectory) > 7:
        print("Warning: Many steps - might be inefficient")
```

## Key Takeaways

1. **Interleave reasoning and acting**: Think, do, observe in a loop.

2. **Ground reasoning in reality**: Observations correct hallucinations.

3. **Traceable decision-making**: Explicit thoughts explain actions.

4. **Recoverable from errors**: Can reason about failures and adapt.

5. **Foundation for agents**: Core pattern for tool-using systems.

6. **Few-shot works well**: 2-3 examples demonstrate the pattern.

7. **Tool design matters**: Clear descriptions and reliable tools are essential.
