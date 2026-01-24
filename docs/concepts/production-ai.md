---
tags:
  - production
  - ai
  - observability
  - concepts
---

# Production AI

What changes when we ship LLM applications to production. Observability, cost, reliability, and operational concerns.

---

## How do we observe LLM applications?

Do we build debuggable systems?

**The challenge**: LLMs are black boxes. When something goes wrong, we need visibility.

**What to log**:

1. **Inputs**: Full prompt, including system message
2. **Outputs**: Complete response, including tool calls
3. **Metadata**: Model, temperature, latency, token count, cost
4. **Context**: Request ID, user ID, session ID for tracing

```python
def call_llm(messages, **kwargs):
    request_id = generate_id()
    start_time = time.time()
    
    response = client.chat.completions.create(
        model=kwargs.get("model", "gpt-4"),
        messages=messages,
        **kwargs
    )
    
    logger.info({
        "event": "llm_call",
        "request_id": request_id,
        "model": response.model,
        "input_tokens": response.usage.prompt_tokens,
        "output_tokens": response.usage.completion_tokens,
        "latency_ms": (time.time() - start_time) * 1000,
        "prompt_preview": messages[-1]["content"][:200],
    })
    
    return response
```

**Tracing for agents**: When agents make multiple calls, we need to trace the full chain:
```
Request → LLM Call #1 → Tool: Search → LLM Call #2 → Response
   └─────────────────────────────────────────────────────┘
                     All linked by trace_id
```

**Tools**: Arize, LangSmith, Weights & Biases, OpenTelemetry with custom spans

---

## What metrics matter for LLM applications?

Do we know how to measure success?

**Operational metrics**:

| Metric | What it tells us |
|--------|------------------|
| Latency (p50, p95, p99) | User experience, timeout risk |
| Token usage | Cost driver |
| Error rate | Reliability |
| Rate limit hits | Capacity issues |

**Quality metrics**:

| Metric | How to measure |
|--------|----------------|
| Task completion rate | Did the user's task succeed? |
| Hallucination rate | Spot checks, automated detection |
| User satisfaction | Thumbs up/down, NPS |
| Answer relevance | LLM-as-judge, human eval |

**RAG-specific metrics**:

| Metric | What it measures |
|--------|------------------|
| Retrieval precision | Are retrieved docs relevant? |
| Answer attribution | Is the answer grounded in sources? |
| Chunk utilization | Are we using the right amount of context? |

**Cost metrics**:
- Cost per request
- Cost per successful task
- Cost trend over time

**Dashboard essentials**:
- Real-time error rates
- Token usage by model
- Latency distribution
- Top failure modes

---

## How do we handle rate limits and retries?

Do we build resilient systems?

**The reality**: LLM APIs have rate limits (requests per minute, tokens per minute). We will hit them.

**Rate limit handling**:

```python
import time
from tenacity import retry, wait_exponential, retry_if_exception_type

class RateLimitError(Exception):
    pass

@retry(
    retry=retry_if_exception_type(RateLimitError),
    wait=wait_exponential(min=1, max=60),
)
def call_llm_with_retry(messages):
    try:
        return client.chat.completions.create(
            model="gpt-4",
            messages=messages,
        )
    except openai.RateLimitError as e:
        raise RateLimitError(str(e))
```

**Strategies**:

1. **Exponential backoff**: Wait 1s, 2s, 4s, 8s... between retries
2. **Jitter**: Add randomness to prevent thundering herd
3. **Queue-based**: Don't call directly; queue requests and process at controlled rate
4. **Fallback models**: If primary is rate-limited, try secondary
5. **Graceful degradation**: Return cached/simpler response if API unavailable

**Headers to watch**:
- `x-ratelimit-remaining`: How many requests left
- `x-ratelimit-reset`: When limits reset
- `Retry-After`: How long to wait (on 429)

**Proactive rate limiting**: Track our own usage, throttle before hitting API limits.

---

## What's the caching strategy for LLM calls?

Do we optimize for cost and latency?

**Why cache**: LLM calls are expensive (time and money). Same prompt → same response (usually).

**Caching layers**:

1. **Exact match**: Hash the prompt, cache the response
   - Fast, simple
   - Miss rate high if prompts vary slightly

2. **Semantic cache**: Embed the query, find similar cached queries
   - Handles paraphrases
   - More complex, possible false matches

3. **Response fragments**: Cache common subcomponents
   - "What's our return policy?" → cache the policy text
   - Compose responses from cached parts

**Implementation**:

```python
import hashlib
from functools import lru_cache

def cache_key(messages: list[dict]) -> str:
    """Create deterministic cache key from messages."""
    content = json.dumps(messages, sort_keys=True)
    return hashlib.sha256(content.encode()).hexdigest()

# In-memory (for development)
response_cache = {}

def cached_llm_call(messages, **kwargs):
    key = cache_key(messages)
    
    if key in response_cache:
        return response_cache[key]
    
    response = call_llm(messages, **kwargs)
    response_cache[key] = response
    return response
```

**Cache invalidation considerations**:
- Model updates may change responses
- Time-sensitive data shouldn't be cached long
- TTL should match our freshness requirements

**When NOT to cache**:
- Personalized responses
- Time-sensitive queries
- When randomness/creativity is desired (high temperature)

---

## How do we A/B test AI features?

Do we ship with data-driven confidence?

**The challenge**: LLM outputs are variable. Traditional A/B testing assumes deterministic behavior.

**Approach**:

1. **Define success metrics**: Task completion, user satisfaction, cost, latency
2. **Randomize assignment**: User/session → variant (A or B)
3. **Collect data**: Same metrics for both variants
4. **Statistical analysis**: Account for LLM variance

**What to test**:
- Different prompts
- Different models
- Different retrieval strategies
- Different chunk sizes
- Agent vs chain architecture

**Example setup**:

```python
import random

def get_variant(user_id: str) -> str:
    """Deterministic variant assignment."""
    hash_val = hash(user_id) % 100
    if hash_val < 50:
        return "control"
    else:
        return "experiment"

def answer_question(user_id: str, question: str) -> str:
    variant = get_variant(user_id)
    
    if variant == "control":
        response = original_prompt_approach(question)
    else:
        response = new_prompt_approach(question)
    
    log_experiment(user_id, variant, question, response)
    return response
```

**Evaluation challenges**:
- Need more samples due to output variance
- Consider LLM-as-judge for quality comparison
- Human evaluation for subset validation

---

## What guardrails should production LLM apps have?

Do we build safe systems?

**Input guardrails**:

1. **Length limits**: Reject excessively long inputs
2. **Content filtering**: Detect prompt injection, harmful content
3. **Rate limiting per user**: Prevent abuse
4. **Input validation**: Expected format, characters

**Output guardrails**:

1. **Content filtering**: Check for harmful/inappropriate content
2. **PII detection**: Catch accidental data leakage
3. **Format validation**: JSON parses correctly, required fields present
4. **Length limits**: Response not excessively long

**Implementation pattern**:

```python
def safe_llm_call(messages):
    # Input guardrails
    if len(messages[-1]["content"]) > MAX_INPUT_LENGTH:
        raise ValueError("Input too long")
    
    if contains_injection_patterns(messages[-1]["content"]):
        return "I can't process that request."
    
    # Call LLM
    response = call_llm(messages)
    
    # Output guardrails
    content = response.choices[0].message.content
    
    if contains_pii(content):
        content = redact_pii(content)
    
    if not passes_content_filter(content):
        return "I can't provide that response."
    
    return content
```

**Defense in depth**: Multiple layers, assume each can fail.

**Monitoring for abuse**: Track patterns that indicate manipulation attempts.

---

## How do we handle cost control?

Can we build economically sustainable systems?

**Cost drivers**:
- Input tokens (prompts, context)
- Output tokens (responses)
- Model choice (GPT-4 >> GPT-3.5)
- Volume

**Cost control strategies**:

1. **Model tiering**:
   ```python
   def choose_model(query_complexity):
       if is_simple_query(query):
           return "gpt-3.5-turbo"  # Cheaper
       else:
           return "gpt-4"  # More capable
   ```

2. **Prompt optimization**: Shorter prompts = fewer tokens
3. **Caching**: Don't pay for the same answer twice
4. **Context pruning**: Include only necessary context in prompts
5. **Output limits**: Set `max_tokens` to prevent runaway responses

**Budget enforcement**:

```python
class BudgetTracker:
    def __init__(self, daily_limit_usd: float):
        self.daily_limit = daily_limit_usd
        self.daily_spend = 0.0
        self.last_reset = date.today()
    
    def check_budget(self, estimated_cost: float) -> bool:
        if date.today() > self.last_reset:
            self.daily_spend = 0.0
            self.last_reset = date.today()
        
        return (self.daily_spend + estimated_cost) <= self.daily_limit
    
    def record_spend(self, actual_cost: float):
        self.daily_spend += actual_cost
```

**Alerting**: Notify when spend exceeds thresholds.

**Per-user limits**: Prevent single user from exhausting budget.

---

## What's the latency budget for LLM features?

Do we understand user experience constraints?

**Reality check**: LLM calls are slow. 1-10 seconds is typical. Users expect faster.

**Latency breakdown** (typical):
- Network to API: 50-200ms
- Queue wait: 0-500ms (during high load)
- Model inference: 500ms-5s (depends on output length)
- Network back: 50-200ms

**Strategies for perceived performance**:

1. **Streaming**: Show tokens as they generate
   - User sees progress immediately
   - Actual latency same, perceived latency much lower

2. **Optimistic UI**: Show placeholder, fill in response
   
3. **Parallel calls**: If multiple LLM calls needed, parallelize
   ```python
   results = await asyncio.gather(
       call_llm(prompt1),
       call_llm(prompt2),
   )
   ```

4. **Smaller models for speed-critical paths**: GPT-3.5 is ~3x faster than GPT-4

5. **Pre-computation**: Generate common responses ahead of time

**Setting expectations**:
- Show loading indicator
- "Thinking..." message
- Progress indication for multi-step processes

**Timeout handling**: If response takes too long, fail gracefully with useful message.

**Architecture consideration**: LLM calls in critical path vs async background processing.
