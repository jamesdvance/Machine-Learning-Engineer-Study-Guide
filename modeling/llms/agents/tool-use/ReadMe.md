# Tool Use for LLM Agents

## Summary

Tool use extends LLM capabilities beyond text generation by enabling agents to interact with external systems, retrieve information, and take actions in the world. Without tools, LLMs are limited to their training data and cannot access current information, execute code, or modify systems. Well-designed tool integration transforms LLMs from conversational interfaces into capable agents that can accomplish real-world tasks.

Key points to remember:

- Tools bridge the gap between LLM knowledge and real-world action
- Tool descriptions are critical: the LLM must understand when and how to use each tool
- Function calling APIs provide structured tool invocation with schema validation
- Error handling and graceful degradation are essential for robustness
- Tool composition enables complex workflows from simple primitives
- Security considerations (sandboxing, permissions) are critical for production
- Observability (logging, tracing) is necessary for debugging and monitoring

## Why Tools Matter

### LLM Limitations Without Tools

| Limitation | Impact | Tool Solution |
|------------|--------|---------------|
| Knowledge cutoff | No current information | Web search, API access |
| No computation | Math errors, no code execution | Calculator, code interpreter |
| No persistence | Forgets between sessions | Database, file system |
| No actions | Cannot affect external world | APIs, system commands |
| No verification | Cannot check its outputs | Testing tools, validators |

### The Tool Use Loop

```
User Query
    |
    v
+-------------------+
| LLM Reasoning     |
| - Understand task |
| - Select tool     |
| - Format call     |
+-------------------+
    |
    v
+-------------------+
| Tool Execution    |
| - Validate params |
| - Execute action  |
| - Return result   |
+-------------------+
    |
    v
+-------------------+
| LLM Processing    |
| - Parse result    |
| - Continue/finish |
+-------------------+
    |
    v
Response or Next Tool Call
```

## Tool Definition

### Basic Tool Structure

```python
from typing import Any, Dict, Optional, Callable
from dataclasses import dataclass, field

@dataclass
class Tool:
    name: str
    description: str
    function: Callable
    parameters: Dict[str, Any]
    required_params: list = field(default_factory=list)
    returns: str = "string"

    def execute(self, **kwargs) -> Any:
        """Execute the tool with given parameters."""
        # Validate required parameters
        missing = [p for p in self.required_params if p not in kwargs]
        if missing:
            raise ValueError(f"Missing required parameters: {missing}")

        return self.function(**kwargs)

    def to_schema(self) -> Dict:
        """Convert to OpenAI function calling schema."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": self.parameters,
                "required": self.required_params
            }
        }
```

### Tool Examples

```python
# Web search tool
search_tool = Tool(
    name="web_search",
    description="Search the web for current information. Use for recent events, facts you're unsure about, or when the user asks for up-to-date information.",
    function=lambda query: search_engine.search(query),
    parameters={
        "query": {
            "type": "string",
            "description": "The search query"
        }
    },
    required_params=["query"]
)

# Code execution tool
code_tool = Tool(
    name="execute_python",
    description="Execute Python code in a sandboxed environment. Use for calculations, data processing, or testing code snippets.",
    function=lambda code: sandbox.execute(code),
    parameters={
        "code": {
            "type": "string",
            "description": "Python code to execute"
        }
    },
    required_params=["code"]
)

# File reading tool
read_tool = Tool(
    name="read_file",
    description="Read the contents of a file. Use when you need to examine file contents.",
    function=lambda path: open(path).read(),
    parameters={
        "path": {
            "type": "string",
            "description": "Path to the file to read"
        }
    },
    required_params=["path"]
)
```

## Function Calling

### OpenAI-Style Function Calling

```python
import openai
import json

class FunctionCallingAgent:
    def __init__(self, model="gpt-4"):
        self.model = model
        self.tools = {}

    def register_tool(self, tool: Tool):
        """Register a tool for the agent to use."""
        self.tools[tool.name] = tool

    def get_functions_schema(self):
        """Get schema for all registered tools."""
        return [tool.to_schema() for tool in self.tools.values()]

    def run(self, user_message: str, max_iterations: int = 10):
        """Run the agent with function calling."""
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=messages,
                functions=self.get_functions_schema(),
                function_call="auto"
            )

            message = response.choices[0].message

            # Check if model wants to call a function
            if message.get("function_call"):
                function_name = message["function_call"]["name"]
                arguments = json.loads(message["function_call"]["arguments"])

                # Execute the function
                result = self._execute_function(function_name, arguments)

                # Add function result to messages
                messages.append(message)
                messages.append({
                    "role": "function",
                    "name": function_name,
                    "content": str(result)
                })
            else:
                # Model is done, return final response
                return message["content"]

        return "Max iterations reached"

    def _execute_function(self, name: str, arguments: Dict) -> Any:
        """Execute a registered function."""
        if name not in self.tools:
            return f"Error: Unknown function {name}"

        try:
            return self.tools[name].execute(**arguments)
        except Exception as e:
            return f"Error executing {name}: {str(e)}"
```

### Anthropic-Style Tool Use

```python
import anthropic

class AnthropicToolAgent:
    def __init__(self, model="claude-3-sonnet-20240229"):
        self.client = anthropic.Anthropic()
        self.model = model
        self.tools = {}

    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool

    def get_tools_schema(self):
        """Get Anthropic-compatible tool schema."""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": {
                    "type": "object",
                    "properties": tool.parameters,
                    "required": tool.required_params
                }
            }
            for tool in self.tools.values()
        ]

    def run(self, user_message: str, max_iterations: int = 10):
        """Run agent with Anthropic tool use."""
        messages = [{"role": "user", "content": user_message}]

        for _ in range(max_iterations):
            response = self.client.messages.create(
                model=self.model,
                max_tokens=4096,
                tools=self.get_tools_schema(),
                messages=messages
            )

            # Check stop reason
            if response.stop_reason == "end_turn":
                # Extract text response
                return self._extract_text(response.content)

            elif response.stop_reason == "tool_use":
                # Execute tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result = self._execute_function(
                            block.name,
                            block.input
                        )
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result)
                        })

                # Add assistant message and tool results
                messages.append({"role": "assistant", "content": response.content})
                messages.append({"role": "user", "content": tool_results})

        return "Max iterations reached"
```

## Tool Description Best Practices

### Writing Effective Descriptions

```python
# BAD: Vague description
bad_tool = Tool(
    name="search",
    description="Searches for things",  # Too vague
    ...
)

# GOOD: Specific with use cases
good_tool = Tool(
    name="web_search",
    description="""Search the web for current information.

Use this tool when:
- You need information about recent events (after your knowledge cutoff)
- You need to verify facts you're unsure about
- The user explicitly asks for current/latest information
- You need real-time data (weather, stock prices, sports scores)

Do NOT use when:
- The information is well-known historical fact
- You're confident in your knowledge
- The query is about your own capabilities""",
    ...
)
```

### Parameter Documentation

```python
good_params = {
    "query": {
        "type": "string",
        "description": "The search query. Be specific and include relevant context. "
                       "Example: 'latest Python 3.12 features' instead of just 'Python'"
    },
    "num_results": {
        "type": "integer",
        "description": "Number of results to return (1-10). Default is 5. "
                       "Use more for research tasks, fewer for quick lookups.",
        "minimum": 1,
        "maximum": 10
    },
    "date_filter": {
        "type": "string",
        "enum": ["day", "week", "month", "year", "any"],
        "description": "Filter results by recency. Use 'day' or 'week' for breaking news."
    }
}
```

## Error Handling

### Robust Tool Execution

```python
from enum import Enum
from dataclasses import dataclass
from typing import Union

class ErrorType(Enum):
    VALIDATION = "validation"
    EXECUTION = "execution"
    TIMEOUT = "timeout"
    PERMISSION = "permission"
    RATE_LIMIT = "rate_limit"

@dataclass
class ToolResult:
    success: bool
    output: Any = None
    error: Optional[str] = None
    error_type: Optional[ErrorType] = None

class RobustTool:
    def __init__(self, tool: Tool, timeout: float = 30.0):
        self.tool = tool
        self.timeout = timeout

    def execute(self, **kwargs) -> ToolResult:
        """Execute with comprehensive error handling."""
        # Validate parameters
        validation_error = self._validate(kwargs)
        if validation_error:
            return ToolResult(
                success=False,
                error=validation_error,
                error_type=ErrorType.VALIDATION
            )

        # Execute with timeout
        try:
            import signal

            def timeout_handler(signum, frame):
                raise TimeoutError(f"Tool execution exceeded {self.timeout}s")

            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.timeout))

            try:
                result = self.tool.execute(**kwargs)
                return ToolResult(success=True, output=result)
            finally:
                signal.alarm(0)

        except TimeoutError as e:
            return ToolResult(
                success=False,
                error=str(e),
                error_type=ErrorType.TIMEOUT
            )
        except PermissionError as e:
            return ToolResult(
                success=False,
                error=f"Permission denied: {e}",
                error_type=ErrorType.PERMISSION
            )
        except Exception as e:
            return ToolResult(
                success=False,
                error=f"Execution failed: {e}",
                error_type=ErrorType.EXECUTION
            )

    def _validate(self, kwargs) -> Optional[str]:
        """Validate parameters against schema."""
        # Check required
        for param in self.tool.required_params:
            if param not in kwargs:
                return f"Missing required parameter: {param}"

        # Check types
        for param, value in kwargs.items():
            if param in self.tool.parameters:
                expected_type = self.tool.parameters[param].get("type")
                if not self._check_type(value, expected_type):
                    return f"Parameter {param} has wrong type"

        return None
```

### Error Recovery Strategies

```python
class ErrorRecoveryAgent:
    def __init__(self, agent, llm):
        self.agent = agent
        self.llm = llm

    def execute_with_recovery(self, tool_name: str, kwargs: Dict) -> ToolResult:
        """Execute tool with automatic error recovery."""
        result = self.agent.tools[tool_name].execute(**kwargs)

        if result.success:
            return result

        # Attempt recovery based on error type
        if result.error_type == ErrorType.VALIDATION:
            # Ask LLM to fix parameters
            fixed_kwargs = self._fix_parameters(tool_name, kwargs, result.error)
            if fixed_kwargs:
                return self.execute_with_recovery(tool_name, fixed_kwargs)

        elif result.error_type == ErrorType.RATE_LIMIT:
            # Wait and retry
            import time
            time.sleep(60)
            return self.execute_with_recovery(tool_name, kwargs)

        elif result.error_type == ErrorType.PERMISSION:
            # Try alternative tool
            alternative = self._find_alternative_tool(tool_name)
            if alternative:
                return self.execute_with_recovery(alternative, kwargs)

        return result

    def _fix_parameters(self, tool_name: str, kwargs: Dict, error: str) -> Optional[Dict]:
        """Use LLM to fix invalid parameters."""
        tool = self.agent.tools[tool_name]
        prompt = f"""The tool call failed with this error: {error}

Tool: {tool.name}
Parameters provided: {kwargs}
Expected schema: {tool.parameters}

Provide corrected parameters as JSON."""

        response = self.llm.generate(prompt)
        try:
            return json.loads(response)
        except:
            return None
```

## Tool Composition

### Chaining Tools

```python
class ToolChain:
    """Chain multiple tools together."""

    def __init__(self, tools: list[Tool]):
        self.tools = tools

    def execute(self, initial_input: Any) -> Any:
        """Execute tools in sequence, passing output to next input."""
        current = initial_input

        for tool in self.tools:
            result = tool.execute(input=current)
            if not result.success:
                raise Exception(f"Chain failed at {tool.name}: {result.error}")
            current = result.output

        return current

# Example: Fetch URL -> Extract text -> Summarize
fetch_chain = ToolChain([
    url_fetcher,      # Returns HTML
    html_to_text,     # Extracts text
    summarizer        # Summarizes text
])
```

### Parallel Tool Execution

```python
import asyncio
from typing import List, Tuple

class ParallelToolExecutor:
    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools

    async def execute_parallel(self, calls: List[Tuple[str, Dict]]) -> List[ToolResult]:
        """Execute multiple tool calls in parallel."""
        tasks = [
            self._execute_async(name, kwargs)
            for name, kwargs in calls
        ]
        return await asyncio.gather(*tasks)

    async def _execute_async(self, name: str, kwargs: Dict) -> ToolResult:
        """Execute a single tool call asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            lambda: self.tools[name].execute(**kwargs)
        )

# Usage
executor = ParallelToolExecutor(tools)
results = await executor.execute_parallel([
    ("web_search", {"query": "Python 3.12"}),
    ("web_search", {"query": "Rust 2024"}),
    ("web_search", {"query": "Go 1.22"})
])
```

### Dynamic Tool Selection

```python
class DynamicToolSelector:
    """Let the LLM choose from available tools."""

    def __init__(self, llm, tools: List[Tool]):
        self.llm = llm
        self.tools = {t.name: t for t in tools}

    def select_and_execute(self, task: str) -> ToolResult:
        """Select appropriate tool for task."""
        tool_descriptions = "\n".join([
            f"- {t.name}: {t.description}"
            for t in self.tools.values()
        ])

        prompt = f"""Given this task: {task}

Available tools:
{tool_descriptions}

Which tool should be used? Respond with just the tool name."""

        selected = self.llm.generate(prompt).strip()

        if selected not in self.tools:
            return ToolResult(
                success=False,
                error=f"Unknown tool: {selected}"
            )

        # Get parameters for selected tool
        params = self._extract_parameters(task, self.tools[selected])

        return self.tools[selected].execute(**params)
```

## Security Considerations

### Sandboxing Code Execution

```python
import subprocess
import tempfile
import os

class SandboxedCodeExecutor:
    """Execute code in isolated environment."""

    def __init__(self, timeout: int = 30, memory_limit_mb: int = 512):
        self.timeout = timeout
        self.memory_limit = memory_limit_mb

    def execute_python(self, code: str) -> ToolResult:
        """Execute Python in a sandboxed subprocess."""
        # Write code to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            script_path = f.name

        try:
            result = subprocess.run(
                [
                    'python', '-u', script_path
                ],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={
                    'PATH': '/usr/bin',
                    'HOME': tempfile.gettempdir(),
                    # Restrict network, filesystem, etc.
                },
                cwd=tempfile.gettempdir()
            )

            if result.returncode == 0:
                return ToolResult(success=True, output=result.stdout)
            else:
                return ToolResult(
                    success=False,
                    error=result.stderr,
                    error_type=ErrorType.EXECUTION
                )

        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                error=f"Execution timed out after {self.timeout}s",
                error_type=ErrorType.TIMEOUT
            )
        finally:
            os.unlink(script_path)
```

### Permission System

```python
from enum import Flag, auto

class Permission(Flag):
    READ_FILES = auto()
    WRITE_FILES = auto()
    EXECUTE_CODE = auto()
    NETWORK_ACCESS = auto()
    SYSTEM_COMMANDS = auto()

class PermissionedToolExecutor:
    def __init__(self, permissions: Permission):
        self.permissions = permissions
        self.tool_permissions = {
            'read_file': Permission.READ_FILES,
            'write_file': Permission.WRITE_FILES,
            'execute_python': Permission.EXECUTE_CODE,
            'web_search': Permission.NETWORK_ACCESS,
            'run_command': Permission.SYSTEM_COMMANDS
        }

    def execute(self, tool: Tool, **kwargs) -> ToolResult:
        """Execute tool if permitted."""
        required = self.tool_permissions.get(tool.name, Permission(0))

        if not (self.permissions & required):
            return ToolResult(
                success=False,
                error=f"Permission denied: {tool.name} requires {required.name}",
                error_type=ErrorType.PERMISSION
            )

        return tool.execute(**kwargs)
```

## Observability

### Tool Logging

```python
import logging
from datetime import datetime
from typing import Dict, Any

class ToolLogger:
    def __init__(self, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(__name__)
        self.history = []

    def log_call(self, tool_name: str, kwargs: Dict, result: ToolResult):
        """Log a tool call with full context."""
        entry = {
            'timestamp': datetime.now().isoformat(),
            'tool': tool_name,
            'parameters': kwargs,
            'success': result.success,
            'output': str(result.output)[:500] if result.output else None,
            'error': result.error,
            'error_type': result.error_type.value if result.error_type else None
        }

        self.history.append(entry)

        if result.success:
            self.logger.info(f"Tool {tool_name} succeeded", extra=entry)
        else:
            self.logger.error(f"Tool {tool_name} failed: {result.error}", extra=entry)

    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregate metrics for tool usage."""
        if not self.history:
            return {}

        by_tool = {}
        for entry in self.history:
            name = entry['tool']
            if name not in by_tool:
                by_tool[name] = {'calls': 0, 'successes': 0, 'failures': 0}

            by_tool[name]['calls'] += 1
            if entry['success']:
                by_tool[name]['successes'] += 1
            else:
                by_tool[name]['failures'] += 1

        return {
            'total_calls': len(self.history),
            'by_tool': by_tool,
            'success_rate': sum(1 for e in self.history if e['success']) / len(self.history)
        }
```

### Tracing for Debugging

```python
class TracedToolExecutor:
    """Tool executor with detailed tracing."""

    def __init__(self, tools: Dict[str, Tool]):
        self.tools = tools
        self.traces = []

    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        trace = {
            'id': len(self.traces),
            'tool': tool_name,
            'input': kwargs,
            'start_time': datetime.now()
        }

        try:
            result = self.tools[tool_name].execute(**kwargs)
            trace['output'] = result
            trace['success'] = result.success
        except Exception as e:
            trace['exception'] = str(e)
            trace['success'] = False
            result = ToolResult(success=False, error=str(e))

        trace['end_time'] = datetime.now()
        trace['duration_ms'] = (trace['end_time'] - trace['start_time']).total_seconds() * 1000

        self.traces.append(trace)
        return result

    def print_trace(self, trace_id: int):
        """Pretty print a trace for debugging."""
        trace = self.traces[trace_id]
        print(f"=== Trace {trace_id} ===")
        print(f"Tool: {trace['tool']}")
        print(f"Input: {trace['input']}")
        print(f"Duration: {trace['duration_ms']:.2f}ms")
        print(f"Success: {trace['success']}")
        if trace['success']:
            print(f"Output: {trace['output']}")
        else:
            print(f"Error: {trace.get('exception', trace['output'].error)}")
```

## Common Tool Categories

### Essential Tools for Agents

| Category | Tools | Purpose |
|----------|-------|---------|
| Information | web_search, wiki_lookup, url_fetch | Get external knowledge |
| Computation | calculator, code_executor, regex_tester | Precise calculations |
| File System | read_file, write_file, list_directory | Persist and access data |
| Communication | send_email, api_call, notify | External interactions |
| Memory | store_memory, retrieve_memory | Long-term persistence |

### Tool Implementation Patterns

```python
# Information retrieval
def make_search_tool(search_api):
    return Tool(
        name="search",
        description="Search for information",
        function=lambda q: search_api.search(q),
        parameters={"query": {"type": "string"}},
        required_params=["query"]
    )

# Stateful tools with context
class StatefulCalculator:
    def __init__(self):
        self.memory = {}

    def calculate(self, expression: str, store_as: str = None) -> float:
        # Replace memory references
        for var, val in self.memory.items():
            expression = expression.replace(var, str(val))

        result = eval(expression)  # In production, use safe_eval

        if store_as:
            self.memory[store_as] = result

        return result
```

## Key Takeaways

1. **Tools enable action**: They transform LLMs from text generators into capable agents.

2. **Descriptions are critical**: The LLM only knows what you tell it about each tool.

3. **Handle errors gracefully**: Tools fail; agents must recover or report clearly.

4. **Security is non-negotiable**: Sandbox code execution, enforce permissions, validate inputs.

5. **Compose for power**: Chain and parallelize tools for complex workflows.

6. **Observe everything**: Logging and tracing are essential for debugging and monitoring.

7. **Match tools to tasks**: Provide the right tools for your agent's domain; too many overwhelm, too few limit.
