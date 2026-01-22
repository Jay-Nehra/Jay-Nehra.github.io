# Agent Architecture from First Principles

AI agents are systems that use LLMs to reason about tasks and take actions autonomously. They go beyond simple question-answering to multi-step problem solving with tool use.

This document explains agent architecture from the ground up: what agents actually are, how the agent loop works, how tool calling functions, and what breaks in production. We'll use the Strands SDK as a concrete example, but the concepts apply to any agent framework.

---

## 1. What an Agent Actually Is

The term "agent" is overloaded. Let's define it precisely.

### Not a Chatbot, Not a Workflow

An agent is not just a chatbot with tools. A chatbot responds to messages. An agent pursues goals.

An agent is not a fixed workflow. Workflows have predetermined steps. Agents decide their steps at runtime based on the situation.

An agent is a system that:
1. Receives a goal or task
2. Reasons about what actions to take
3. Executes actions (tool calls)
4. Observes results
5. Decides what to do next
6. Repeats until the goal is achieved (or it gives up)

The key distinction is **autonomy in decision-making**. The agent decides what to do, not a predetermined script.

### The Observe-Think-Act Loop

Agents follow a loop:

**Observe:** Take in information (user request, tool results, environment state)

**Think:** Reason about what to do next (this is where the LLM runs)

**Act:** Execute an action (call a tool, respond to user, update state)

This loop repeats until the agent decides it's done.

The loop is not fixed-length. An agent might complete a task in one iteration or twenty. It might call the same tool multiple times or use different tools each time. The LLM makes these decisions dynamically.

### Why Agents Exist

Agents exist because some tasks cannot be decomposed into fixed steps ahead of time.

**Fixed workflow works when:**
- Steps are known in advance
- Same process works for all inputs
- No decision-making needed during execution

**Agents work when:**
- Steps depend on intermediate results
- Different inputs require different approaches
- Decisions must be made at runtime

Example: "Summarize this document" is a fixed workflow. "Research this topic and write a report" is an agent task — the research path depends on what's found.

### The Fundamental Tradeoff: Autonomy vs Control

More autonomy means:
- More flexible problem-solving
- Less predictable behavior
- Higher risk of unexpected actions
- Harder to debug

More control means:
- More predictable behavior
- Less flexibility
- More engineering effort to handle edge cases
- Easier to reason about

Most production agents sit somewhere in the middle: constrained autonomy with guardrails.

---

## 2. The Agent Loop Execution Model

Understanding the agent loop in detail helps you design robust agents and debug failures.

### Receive Input

The agent receives a task. This could be:
- A user message: "Book me a flight to New York next Tuesday"
- A system trigger: "Process incoming order #12345"
- Another agent's request: "Research this subtopic for me"

The input becomes part of the agent's context.

### Reason About Next Action

The LLM is called with:
- System prompt (agent's instructions and personality)
- Available tools (function schemas)
- Conversation history (previous turns, tool results)
- Current input

The LLM generates one of:
- A text response (agent is done or communicating)
- A tool call request (agent wants to take an action)

This is where the "thinking" happens. The model decides what to do based on all available context.

### Call Tool (Or Respond)

If the LLM requested a tool call, you execute it:

```python
tool_call = response.tool_calls[0]
function_name = tool_call.function.name
arguments = json.loads(tool_call.function.arguments)

result = execute_tool(function_name, arguments)
```

The tool result is added to the conversation history.

If the LLM responded with text and no tool calls, the agent is done (or awaiting user input).

### Observe Result

The tool result becomes the next observation. This might be:
- Data: "Current weather in NYC: 72°F, sunny"
- Confirmation: "Email sent successfully"
- Error: "API rate limit exceeded, try again in 60 seconds"
- Complex output: JSON data, file contents, database results

### Repeat Until Done

The loop continues:
1. Tool result added to history
2. LLM called again with updated context
3. LLM decides next action (another tool, or final response)
4. Execute, observe, repeat

### What "Done" Means

The agent is done when:
- It responds with text (no tool calls)
- It explicitly signals completion
- It hits a maximum iteration limit
- An error forces termination

Determining "done" is non-trivial. The LLM might think it's done when it isn't. Or it might keep trying when it should stop.

---

## 3. Tool Calling Mechanics

Tools are how agents interact with the world. Understanding tool calling in detail is essential.

### How Function Schemas Work

Tools are defined as JSON schemas:

```python
{
    "type": "function",
    "function": {
        "name": "search_web",
        "description": "Search the web for information",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query"
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
}
```

This schema tells the model:
- What the tool does (description)
- What parameters it accepts (properties)
- What's required vs optional
- Parameter types and constraints

The schema is included in every LLM call where the tool is available.

### What the Model Sees

When the LLM processes a request with tools, it sees:
- The system prompt
- The conversation history
- The tool schemas (as structured data)

The model is trained to understand these schemas and generate valid tool calls.

### What the Model Returns

When the model decides to call a tool, it returns:

```python
{
    "role": "assistant",
    "content": null,
    "tool_calls": [
        {
            "id": "call_abc123",
            "type": "function",
            "function": {
                "name": "search_web",
                "arguments": "{\"query\": \"best flights NYC\", \"num_results\": 3}"
            }
        }
    ]
}
```

The model does not execute the function. It generates a structured request.

### Your Responsibility: Execute and Return Result

You must:

1. **Parse the tool call:** Extract function name and arguments
2. **Validate arguments:** The model can make mistakes
3. **Execute the function:** Your actual implementation
4. **Handle errors:** Catch exceptions, format error messages
5. **Return result:** Add to conversation as a tool message

```python
# Receive tool call from model
tool_call = response.tool_calls[0]

# Parse
function_name = tool_call.function.name
try:
    arguments = json.loads(tool_call.function.arguments)
except json.JSONDecodeError:
    result = {"error": "Invalid JSON arguments"}
    
# Validate
if function_name not in available_functions:
    result = {"error": f"Unknown function: {function_name}"}
elif not validate_arguments(function_name, arguments):
    result = {"error": "Invalid arguments"}
else:
    # Execute
    try:
        result = available_functions[function_name](**arguments)
    except Exception as e:
        result = {"error": str(e)}

# Return to model
messages.append({"role": "assistant", "content": None, "tool_calls": [tool_call]})
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,
    "content": json.dumps(result)
})
```

### Execution Model of a Tool Call

The complete cycle for one tool call:

1. LLM generates tool call request
2. Your code parses the request
3. Your code validates arguments
4. Your code executes the function
5. Your code formats the result
6. Result is added to conversation
7. LLM is called again with updated history
8. LLM either calls another tool or responds

Each tool call is a full LLM round-trip. This adds latency and cost.

---

## 4. State Management Across Turns

Agents maintain state across multiple turns. Managing this state correctly is crucial.

### Conversation History as State

The simplest state is conversation history — all messages exchanged.

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant..."},
    {"role": "user", "content": "Book me a flight to NYC"},
    {"role": "assistant", "tool_calls": [...]},
    {"role": "tool", "content": "{\"flights\": [...]}"},
    {"role": "assistant", "content": "I found several options..."},
    # ... continues
]
```

Each turn adds messages. The full history is sent to the LLM each time.

### Working Memory vs Long-Term Memory

**Working memory:** Current conversation, recent tool results, active task context. Held in the message history.

**Long-term memory:** Persistent information across sessions. User preferences, past interactions, learned facts. Stored externally (database, vector store).

Most agent frameworks focus on working memory. Long-term memory requires additional infrastructure.

### Context Window Pressure

As conversations grow, context fills up:
- System prompt: 500-2000 tokens (fixed)
- Tool schemas: 500-2000 tokens (fixed per tool)
- Conversation history: grows unbounded
- Reserved for output: 500-2000 tokens

A 10-turn conversation with tool calls can easily reach 10,000+ tokens.

When context fills:
- New turns can't be processed
- Old turns must be dropped or summarized
- Agent loses access to earlier context

### Summarization Strategies

To manage growing context:

**Sliding window:** Keep only the last N messages.
- Simple but loses important early context
- Good for short tasks

**Summarization:** Periodically summarize old messages.
```python
if token_count(messages) > THRESHOLD:
    old_messages = messages[1:-5]  # Keep system prompt and recent
    summary = summarize(old_messages)
    messages = [
        messages[0],  # System prompt
        {"role": "system", "content": f"Previous context: {summary}"},
        *messages[-5:]  # Recent messages
    ]
```

**Hierarchical:** Maintain summaries at different granularities.
- Message-level
- Task-level
- Session-level

### What Breaks with Long Conversations

**Context overflow:** Agent can't process new input.

**Lost context:** Summarization drops important details. Agent forgets earlier instructions or results.

**Cost explosion:** Longer contexts = more tokens = higher cost per turn.

**Degraded quality:** Some models perform worse with very long contexts ("lost in the middle").

Plan for context management from the start.

---

## 5. Strands SDK: Concrete Implementation

Let's make this concrete with the Strands SDK. These patterns apply to other frameworks (LangChain, LlamaIndex, etc.).

### Agent Class Structure

In Strands, an agent is defined with:

```python
from strands import Agent
from strands.tools import tool

@tool
def search_database(query: str) -> dict:
    """Search the product database."""
    return database.search(query)

@tool
def send_email(to: str, subject: str, body: str) -> dict:
    """Send an email to a customer."""
    return email_service.send(to, subject, body)

agent = Agent(
    model="us.anthropic.claude-sonnet-4-20250514",
    system_prompt="You are a customer service agent...",
    tools=[search_database, send_email]
)
```

The `@tool` decorator extracts the function signature and docstring to create the schema automatically.

### Tool Definition Patterns

**Simple tools:** Single function, straightforward parameters.

```python
@tool
def get_weather(city: str) -> dict:
    """Get current weather for a city."""
    return weather_api.get(city)
```

**Complex tools:** Multiple parameters, validation needed.

```python
@tool
def create_order(
    product_id: str,
    quantity: int,
    shipping_address: str,
    express: bool = False
) -> dict:
    """Create a new product order.
    
    Args:
        product_id: The product SKU
        quantity: Number of units (1-100)
        shipping_address: Full shipping address
        express: Whether to use express shipping
    """
    if quantity < 1 or quantity > 100:
        return {"error": "Quantity must be 1-100"}
    return orders.create(product_id, quantity, shipping_address, express)
```

**Async tools:** For I/O-bound operations.

```python
@tool
async def fetch_data(url: str) -> dict:
    """Fetch data from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        return response.json()
```

### Session and Conversation Management

Strands manages conversation state:

```python
# Create a session
session = agent.create_session()

# First turn
response1 = await session.send("What's the weather in NYC?")
# Agent calls get_weather, returns result

# Second turn (same session, remembers context)
response2 = await session.send("What about tomorrow?")
# Agent knows we're talking about NYC weather

# New session (fresh context)
new_session = agent.create_session()
```

Sessions encapsulate conversation history and working state.

### Hooks and Lifecycle Events

Strands provides hooks for observing and controlling the agent loop:

```python
from strands import Agent, Hooks

class MyHooks(Hooks):
    async def on_tool_start(self, tool_name: str, arguments: dict):
        print(f"Calling tool: {tool_name}")
        # Could log, validate, or modify
        
    async def on_tool_end(self, tool_name: str, result: dict):
        print(f"Tool result: {result}")
        
    async def on_agent_end(self, response: str):
        print(f"Agent finished: {response}")

agent = Agent(
    model="...",
    tools=[...],
    hooks=MyHooks()
)
```

Hooks enable:
- Logging and observability
- Cost tracking
- Rate limiting
- Security checks

### Execution Model of a Strands Agent

When you call `session.send(message)`:

1. Message added to conversation history
2. System prompt + tools + history sent to LLM
3. LLM response parsed
4. If tool calls:
   - `on_tool_start` hook called
   - Tool executed
   - `on_tool_end` hook called
   - Result added to history
   - Loop back to step 2
5. If text response:
   - `on_agent_end` hook called
   - Response returned to caller

This is the agent loop made concrete.

---

## 6. Error Handling in Agents

Agents fail in unique ways. Robust error handling is essential.

### Tool Execution Failures

Tools can fail:
- External API errors
- Invalid inputs
- Timeouts
- Rate limits
- Permission denied

Handle failures gracefully:

```python
@tool
def query_api(endpoint: str) -> dict:
    """Query an external API."""
    try:
        response = requests.get(endpoint, timeout=10)
        response.raise_for_status()
        return {"success": True, "data": response.json()}
    except requests.Timeout:
        return {"success": False, "error": "Request timed out"}
    except requests.HTTPError as e:
        return {"success": False, "error": f"HTTP error: {e.response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}
```

Return structured errors so the agent can reason about them and try alternatives.

### Model Refusal

Sometimes the model refuses to continue:
- Content policy violations
- Instructions it interprets as harmful
- Confusion about what to do

Handle refusals:

```python
if "I cannot" in response or "I'm not able to" in response:
    # Model refused
    log_refusal(response)
    return fallback_response()
```

Consider rephrasing requests or using different models.

### Infinite Loop Detection

Agents can get stuck:
- Calling the same tool repeatedly
- Alternating between two tools without progress
- Unable to complete the task but not stopping

Implement safeguards:

```python
MAX_ITERATIONS = 20
SAME_TOOL_LIMIT = 3

iteration = 0
same_tool_count = 0
last_tool = None

while iteration < MAX_ITERATIONS:
    response = await call_llm()
    
    if response.tool_calls:
        tool = response.tool_calls[0].function.name
        
        if tool == last_tool:
            same_tool_count += 1
            if same_tool_count >= SAME_TOOL_LIMIT:
                # Break the loop
                force_completion()
                break
        else:
            same_tool_count = 0
            last_tool = tool
            
    iteration += 1
```

### Graceful Degradation

When the agent can't complete the task:
- Return partial results
- Explain what was accomplished and what failed
- Suggest manual steps the user could take

```python
try:
    result = await agent.complete_task(request)
except AgentFailure as e:
    return {
        "status": "partial",
        "completed_steps": e.completed_steps,
        "error": str(e),
        "suggestion": "You may need to manually complete step X"
    }
```

### What Breaks and How to Recover

| Failure | Detection | Recovery |
|---------|-----------|----------|
| Tool timeout | Timeout exception | Return error, agent retries or uses alternative |
| API rate limit | 429 response | Wait and retry, or switch to fallback |
| Invalid arguments | Validation failure | Return error message, agent corrects |
| Model refusal | Keyword detection | Log, fallback, or rephrase |
| Infinite loop | Iteration/tool counter | Force completion or escalate |
| Context overflow | Token count | Summarize and continue |

---

## 7. Multi-Agent Patterns

Sometimes one agent isn't enough. Multi-agent systems coordinate multiple specialized agents.

### When Single Agent Fails

Single agents struggle with:
- Tasks requiring multiple areas of expertise
- Very long tasks that exceed context
- Tasks requiring parallel work
- Complex coordination

Signs you need multiple agents:
- System prompt is very long (trying to do everything)
- Frequent context overflow
- Poor performance on sub-tasks
- Need for parallelization

### Orchestrator Pattern

One "orchestrator" agent coordinates multiple "worker" agents:

```python
orchestrator = Agent(
    system_prompt="You coordinate a team of specialized agents...",
    tools=[
        delegate_to_researcher,
        delegate_to_writer,
        delegate_to_reviewer,
        compile_final_result
    ]
)

@tool
async def delegate_to_researcher(query: str) -> dict:
    """Delegate research task to the researcher agent."""
    return await researcher_agent.complete(query)

@tool
async def delegate_to_writer(outline: str, research: str) -> dict:
    """Delegate writing task to the writer agent."""
    return await writer_agent.complete(f"Write based on:\n{outline}\n{research}")
```

The orchestrator decides what to delegate. Workers are specialized and focused.

### Peer-to-Peer Coordination

Agents communicate directly:

```python
@tool
async def ask_expert(agent_name: str, question: str) -> dict:
    """Ask another agent for their expertise."""
    expert = agents[agent_name]
    return await expert.complete(question)
```

This is more flexible but harder to control. Agents might:
- Ask each other in circles
- Disagree on approaches
- Duplicate work

### Shared vs Isolated State

**Shared state:** All agents read/write common memory.
- Enables coordination
- Risk of conflicts
- Harder to debug

**Isolated state:** Each agent has its own memory.
- Simpler to reason about
- Requires explicit handoffs
- Less coordination overhead

Most production systems use hybrid: shared read, isolated write with explicit sync points.

### Complexity Tradeoffs

Multi-agent systems add:
- Latency (more LLM calls)
- Cost (more tokens)
- Complexity (more failure modes)
- Debugging difficulty (distributed state)

Use multi-agent when:
- Single agent demonstrably fails
- Task naturally decomposes
- Specialization improves quality
- You can afford the overhead

Start with one agent. Add more only when needed.

---

## 8. What Breaks in Production

Agent systems have unique production challenges.

### Infinite Loops

The most common production failure. The agent keeps calling tools without making progress.

Causes:
- Unclear stopping conditions
- Ambiguous instructions
- Tool returning unhelpful results
- Model confusion

Prevention:
- Hard iteration limits
- Same-tool detection
- Timeout per task
- Clear completion criteria in prompts

### Tool Abuse

The model calls tools incorrectly:
- Wrong tool for the task
- Incorrect arguments
- Calling dangerous tools inappropriately

Prevention:
- Validate arguments thoroughly
- Implement permission systems
- Log all tool calls
- Review before execution for sensitive operations

### Context Explosion

Long-running agents accumulate context until they fail.

Prevention:
- Proactive summarization
- Task decomposition (break into subtasks with fresh context)
- Context monitoring and alerts
- Hard limits with graceful degradation

### Cost Explosion

Agents can be expensive:
- Each iteration is an LLM call
- Long contexts increase per-call cost
- Retries and failures still cost money

Prevention:
- Per-task cost limits
- Monitor and alert on spend
- Use cheaper models for subtasks
- Cache tool results where applicable

### Debugging Difficulty

Agents are non-deterministic. The same input might produce different outputs.

Challenges:
- Hard to reproduce bugs
- State spreads across many turns
- Tool interactions are complex

Solutions:
- Comprehensive logging (full conversation history)
- Session replay capability
- Structured traces (not just text logs)
- Deterministic modes for testing (seed the model)

---

## Summary: The Agent Mental Model

Agents are systems that use LLMs to reason about tasks and take actions autonomously.

**The agent loop:** Observe → Think → Act → Repeat until done.

**Tools** let agents interact with the world. You define schemas; the model generates calls; you execute them.

**State management** is critical. Conversations grow. Context overflows. Plan for summarization.

**Error handling** must be comprehensive. Tools fail. Models refuse. Loops happen.

**Multi-agent** adds power and complexity. Use when single agents fail.

**Production** requires guardrails: iteration limits, cost caps, comprehensive logging.

---

## Interview Framing

**"What is an AI agent and how is it different from a chatbot?"**

"An agent is a system that pursues goals autonomously by deciding what actions to take at runtime. Unlike a chatbot that just responds to messages, an agent can call tools, observe results, and iterate until a task is complete. The key difference is decision-making: the agent decides what to do based on the situation, not following a predetermined script."

**"How does tool calling work in agents?"**

"Tools are defined as JSON schemas that describe what they do and what parameters they accept. When the LLM is called, it sees these schemas and can choose to call a tool by generating a structured request with the function name and arguments. Your code then executes the actual function, returns the result, and the LLM is called again with the updated context. This loop continues until the agent finishes."

**"What are the main challenges with agents in production?"**

"Infinite loops are the most common — the agent keeps calling tools without making progress. You need iteration limits and same-tool detection. Cost control is critical because each iteration is an LLM call. Context management matters because long conversations overflow the context window. And debugging is hard because agents are non-deterministic — you need comprehensive logging and session replay."

**"When would you use multiple agents instead of one?"**

"When a single agent fails — usually because the task requires multiple areas of expertise that don't fit in one system prompt, or because the task is too long and overflows context. The tradeoff is complexity: more agents mean more LLM calls, more cost, and harder debugging. Start with one agent and only add more when you have evidence the single agent can't handle the task."
