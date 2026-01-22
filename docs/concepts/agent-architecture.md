---
tags:
  - agents
  - llm
  - ai
  - concepts
---

# Agent Architecture

How AI agents work—tools, memory, orchestration patterns. Relevant for LangChain, Strands, AgentCore, and similar frameworks.

---

## What is an "agent" vs a "chain"?

**What they're really asking**: Do we understand the architectural distinction?

**Chain**: A fixed sequence of operations. Deterministic flow.
```
Input → Prompt Template → LLM → Parser → Output
```
The path is known at design time. Every input follows the same steps.

**Agent**: An LLM that decides what to do next. Dynamic flow.
```
Input → LLM decides: "I need to search" → Tool: Search → 
LLM decides: "Now I should calculate" → Tool: Calculator →
LLM decides: "I have enough info" → Output
```
The path is determined at runtime by the LLM.

**The key difference**: In a chain, *we* define control flow. In an agent, *the LLM* defines control flow.

**When to use chains**:
- Well-defined tasks with predictable steps
- When we need guarantees about execution
- When we can't afford unpredictable behavior

**When to use agents**:
- Tasks requiring dynamic decision-making
- When the solution path varies by input
- When we trust the LLM to navigate

**Hybrid approach**: Chains that call agents for specific subtasks, or agents with constrained tool sets.

---

## How does tool calling work?

**What they're really asking**: Do we understand the function calling interface?

**The mechanism**:

1. We define tools with names, descriptions, and parameters (JSON schema)
2. We send the user query plus tool definitions to the LLM
3. LLM responds with either:
   - A text response (no tool needed), OR
   - A tool call: `{"name": "search", "arguments": {"query": "..."}}`
4. We execute the tool and return results to the LLM
5. LLM generates final response using tool results

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"}
                },
                "required": ["location"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "What's the weather in London?"}],
    tools=tools,
)

# LLM returns: tool_calls=[{name: "get_weather", arguments: {"location": "London"}}]
```

**Key insight**: The LLM doesn't execute tools. It *decides* which tool to call and with what arguments. We execute and return results.

**Common patterns**:
- Single tool call → execute → respond
- Multiple parallel tool calls → execute all → respond
- Sequential: tool → result → another tool → result → respond (agentic loop)

---

## What's the ReAct pattern?

**What they're really asking**: Do we know the foundational agent architecture?

**ReAct = Reasoning + Acting**

The LLM alternates between:
- **Thought**: Reasoning about what to do next
- **Action**: Choosing a tool to execute
- **Observation**: Processing the tool's result

```
User: What's the population of the capital of France?

Thought: I need to find the capital of France first.
Action: search("capital of France")
Observation: Paris is the capital of France.

Thought: Now I need to find the population of Paris.
Action: search("population of Paris")
Observation: The population of Paris is approximately 2.1 million.

Thought: I now have enough information to answer.
Answer: The population of Paris, the capital of France, is approximately 2.1 million.
```

**Why it works**: Making the LLM verbalize its reasoning improves decision quality and makes debugging easier.

**Variations**:
- **ReAct**: Interleave thought/action/observation
- **Plan-and-Execute**: Plan all steps first, then execute
- **Reflexion**: Self-critique and retry on failures

**Framework implementations**: LangChain's AgentExecutor, Strands' agent loops, OpenAI's function calling loop.

---

## How do we handle agent failures gracefully?

**What they're really asking**: Do we build robust systems?

**Failure modes**:

1. **Tool execution fails**: API down, timeout, invalid response
2. **LLM hallucinates tool calls**: Calls nonexistent tool or wrong arguments
3. **Infinite loops**: Agent keeps calling tools without progress
4. **Context overflow**: Conversation grows too long

**Defensive patterns**:

```python
# 1. Tool-level error handling
def safe_tool_call(tool_name, args, max_retries=3):
    for attempt in range(max_retries):
        try:
            return execute_tool(tool_name, args)
        except ToolError as e:
            if attempt == max_retries - 1:
                return f"Error: {tool_name} failed - {e}"
            time.sleep(2 ** attempt)

# 2. Iteration limits
MAX_ITERATIONS = 10
for i in range(MAX_ITERATIONS):
    response = agent.step()
    if response.is_final:
        break
else:
    return "Agent exceeded maximum iterations"

# 3. Validate tool calls before execution
if tool_call.name not in valid_tools:
    return "Unknown tool requested"

# 4. Timeout the entire agent
with timeout(seconds=60):
    result = agent.run(query)
```

**Graceful degradation**: When an agent fails, fall back to:
- Simpler chain-based approach
- Pre-canned responses
- Human escalation

---

## What's the memory problem in agents?

**What they're really asking**: Do we understand state management in LLM applications?

**The problem**: LLMs are stateless. Each API call starts fresh. How do we maintain conversation history, learned facts, or long-term context?

**Types of memory**:

1. **Conversation history** (short-term):
   - Include previous messages in each prompt
   - Simple but limited by context window
   
2. **Summary memory**:
   - Periodically summarize old messages
   - Trade-off: loses detail but extends effective memory

3. **Entity memory**:
   - Extract and track entities mentioned (people, products, etc.)
   - "User's name is Jay, they prefer Python"

4. **Vector store memory** (long-term):
   - Store past interactions as embeddings
   - Retrieve relevant past context when needed

**AgentCore Memory**: AWS provides managed memory that handles:
- Session-based conversation tracking
- Cross-session retrieval
- Automatic summarization

**Design consideration**: Memory is a RAG problem. What we "remember" is what we choose to retrieve.

---

## How does MCP (Model Context Protocol) fit in?

**What they're really asking**: Do we understand the emerging tool ecosystem?

**The problem MCP solves**: Every application has its own way of exposing tools to LLMs. No standard protocol.

**MCP provides**:
- Standard format for tool definitions
- Standard protocol for tool execution
- Interoperability between tool providers and LLM frameworks

**The architecture**:
```
LLM Framework (Claude, LangChain, Strands)
        ↓
    MCP Client
        ↓
    MCP Server (exposes tools)
        ↓
    Actual Service (database, API, etc.)
```

**Why it matters for us**:
- Write tools once, use with any MCP-compatible framework
- Access ecosystem of pre-built MCP servers
- Standard way to expose our APIs as agent tools

**AgentCore Gateway**: Turns existing APIs into MCP-compatible tool servers.

**Practical impact**: Instead of building custom integrations for each framework, we implement MCP once.

---

## When should we use agents vs deterministic workflows?

**What they're really asking**: Do we have good judgment about architecture?

**Use deterministic workflows when**:
- The task has a known, fixed structure
- We need guaranteed execution paths
- Latency/cost must be predictable
- Failures must be handled in specific ways
- Compliance/audit requires explainability

**Use agents when**:
- The task requires dynamic decision-making
- User intents vary significantly
- We can't anticipate all paths at design time
- Some autonomy is acceptable and valuable

**The spectrum**:
```
Fully Deterministic          Hybrid               Fully Agentic
       ↓                        ↓                       ↓
Fixed chain of     →    Chain with agent    →    LLM decides
operations              for one complex step     everything
```

**Real-world pattern**: Most production systems are hybrid:
- Deterministic orchestration for the main flow
- Agent for specific complex subtasks (like search or reasoning)
- Fallbacks if agent misbehaves

**Cost consideration**: Agents often require multiple LLM calls. If budget is constrained, lean deterministic.

---

## How do we test agent-based systems?

**What they're really asking**: Do we build production-quality AI?

**The challenge**: Agents are non-deterministic. Same input can produce different tool sequences.

**Testing layers**:

1. **Unit tests: Individual tools**
   ```python
   def test_search_tool():
       result = search_tool("Python documentation")
       assert "python.org" in result.lower()
   ```

2. **Integration tests: Tool calling works**
   ```python
   def test_agent_can_use_tools():
       agent = Agent(tools=[search_tool])
       # Mock LLM response to force tool call
       response = agent.run("Search for X")
       assert agent.tool_call_count > 0
   ```

3. **Behavioral tests: End-to-end outcomes**
   ```python
   def test_agent_answers_correctly():
       result = agent.run("What is 2+2?")
       assert "4" in result
   ```

4. **Regression tests: Capture expected behavior**
   - Store (input, expected_output) pairs
   - Run periodically, flag regressions
   - Accept some variation (fuzzy matching)

5. **Evaluation sets: Benchmark quality**
   - Curated set of queries with expected answers
   - Track metrics over time

**Mocking strategy**: Mock the LLM responses to get deterministic tests, then have a smaller set of true end-to-end tests.

**Observability**: Log every tool call, LLM interaction, and decision. Essential for debugging production issues.
