# System Design

Scenario-based design questions for AI-powered features. How we'd approach designing these systems.

---

## Design a customer support chatbot with knowledge base

**The scenario**: We need a chatbot that can answer customer questions using our company's documentation, FAQs, and policy documents.

### Requirements clarification

Before designing, we'd ask:
- How many documents? (100s vs 100,000s affects architecture)
- Update frequency? (Real-time vs daily batch)
- User volume? (10 req/min vs 10,000 req/min)
- Accuracy requirements? (Can we tolerate some wrong answers?)
- Escalation path? (What happens when bot can't answer?)

### High-level architecture

```
User Query
    ↓
[Guardrails: input validation, injection detection]
    ↓
[Query Embedding]
    ↓
[Vector Search] ← Document Chunks (pre-embedded)
    ↓
[Reranking: pick best chunks]
    ↓
[LLM: generate answer with retrieved context]
    ↓
[Guardrails: output validation, PII check]
    ↓
Response + Source Citations
```

### Key components

**1. Document ingestion pipeline**:
- Ingest documents (PDF, HTML, Markdown)
- Chunk into 500-1000 token pieces with overlap
- Embed each chunk
- Store in vector database with metadata (source, date, section)

**2. Query processing**:
- Embed user query
- Search vector DB for top-k similar chunks
- Rerank using cross-encoder for precision
- Select top 3-5 chunks for context

**3. Generation**:
- System prompt: "Answer based only on provided context. If uncertain, say so."
- Include retrieved chunks in prompt
- Generate response with source attribution

**4. Conversation management**:
- Track conversation history
- Include relevant history in subsequent queries
- Summarize long conversations

### Design decisions

**Why RAG over fine-tuning?**
- Documents change frequently
- Need source attribution
- Easier to update without retraining

**Why reranking?**
- Embedding search has false positives
- Cross-encoder is more accurate but slower
- Use embedding for broad recall, reranker for precision

**Handling "I don't know":**
- Prompt engineering to admit uncertainty
- Fallback to human when confidence is low
- Track unanswered questions for content gaps

### Scaling considerations

- **Vector DB**: Managed service (Pinecone, Weaviate) for scale
- **Caching**: Cache frequent queries
- **Async processing**: Queue-based for high volume
- **Model tiering**: Simple questions → smaller model

---

## Design an AI code review assistant

**The scenario**: A tool that reviews pull requests and provides feedback on code quality, potential bugs, and style.

### Requirements clarification

- Scope: What languages? Just Python, or polyglot?
- Integration: GitHub/GitLab webhook? IDE plugin?
- Feedback type: Comments inline? Summary? Both?
- Latency tolerance: Background processing OK?

### High-level architecture

```
PR Webhook
    ↓
[Event Handler: extract diff, files changed]
    ↓
[Context Builder: gather relevant code context]
    ↓
[LLM: analyze code, generate feedback]
    ↓
[Post-processor: format comments, filter noise]
    ↓
Post Comments to PR
```

### Key components

**1. Diff extraction**:
- Parse PR diff to get changed files and line ranges
- Fetch full file content for context
- Identify changed functions/classes

**2. Context building**:
- Include: changed code, surrounding context
- Include: related files (imports, tests)
- Include: repo conventions (from CONTRIBUTING.md, existing code style)

**3. Review generation**:
```python
prompt = f"""
Review this code change. Focus on:
- Bugs or logic errors
- Security issues
- Performance concerns
- Readability improvements

Code style for this repo:
{style_guide}

Changed code:
{diff}

Full file context:
{file_content}
"""
```

**4. Comment placement**:
- Map LLM feedback to specific line numbers
- Post as inline comments on PR
- Aggregate minor issues into summary comment

### Design decisions

**Why not review entire codebase?**
- Context window limits
- Most relevant: what changed + immediate context
- Full codebase analysis is separate tool

**Handling false positives**:
- Confidence thresholds: only comment when confident
- User feedback: thumbs up/down on comments
- Learn from dismissals

**Async processing**:
- Code review can take 30-60 seconds
- Trigger on PR event, post results when ready
- Don't block PR creation

### Scaling considerations

- Queue-based processing for concurrent PRs
- Cache analysis for unchanged files
- Rate limit per repo to control costs

---

## Design a document Q&A system for enterprise

**The scenario**: Employees can ask questions about company policies, procedures, and documentation across multiple sources (Confluence, SharePoint, internal wikis).

### Requirements clarification

- Data sources: How many? What formats?
- Access control: Different permissions by document?
- Update frequency: Real-time sync vs batch?
- Query volume: How many queries per day?
- Compliance: Audit logging required?

### High-level architecture

```
                    ┌─────────────────────────────────────┐
                    │         Document Sources            │
                    │  Confluence | SharePoint | Wiki     │
                    └────────────────┬────────────────────┘
                                     ↓
                    ┌────────────────────────────────────┐
                    │        Ingestion Pipeline          │
                    │  Crawl → Extract → Chunk → Embed   │
                    └────────────────┬───────────────────┘
                                     ↓
                    ┌────────────────────────────────────┐
                    │          Vector Database           │
                    │   Chunks + Metadata + Permissions  │
                    └────────────────┬───────────────────┘
                                     ↑
User Query → [Auth] → [Embed] → [Search with ACL filter] → [LLM] → Response
```

### Key components

**1. Multi-source ingestion**:
- Connectors for each source (Confluence API, SharePoint API)
- Extract text from various formats (HTML, PDF, DOCX)
- Preserve metadata: source, author, last updated, permissions

**2. Access control**:
- Store document permissions with chunks
- At query time, filter to documents user can access
- Never return content user shouldn't see

```python
def search_with_acl(query_embedding, user_groups):
    return vector_db.search(
        embedding=query_embedding,
        filter={
            "allowed_groups": {"$in": user_groups}
        },
        top_k=10
    )
```

**3. Freshness management**:
- Track document versions
- Re-index when documents change
- Handle deleted documents (remove from index)

**4. Answer generation**:
- Ground answers in retrieved documents
- Always cite sources
- Flag when sources are outdated

### Design decisions

**Why per-chunk ACL?**
- Documents may have different access levels
- Query-time filtering ensures security
- More granular than document-level

**Handling conflicting information**:
- Prefer more recent documents
- Surface multiple sources if they differ
- Let user choose authoritative source

**Audit trail**:
- Log every query and response
- Log which documents were retrieved
- Retention for compliance

### Scaling considerations

- Incremental re-indexing (not full rebuild)
- Distributed vector search for large corpora
- Query caching with user-aware cache keys

---

## Design a content moderation pipeline

**The scenario**: User-generated content needs moderation before publishing. Detect and handle harmful content at scale.

### Requirements clarification

- Content types: Text only? Images? Video?
- Volume: How many items per day?
- Latency: Real-time or async?
- Action types: Block, flag for review, allow?
- Categories: What's considered harmful?

### High-level architecture

```
Content Submission
        ↓
[Rule-based pre-filter: obvious violations, spam]
        ↓
[ML classifier: category detection]
        ↓
[LLM review: nuanced cases]
        ↓
Decision Router:
  - Clear violation → Block
  - Likely OK → Allow
  - Uncertain → Human review queue
```

### Key components

**1. Multi-stage pipeline**:

Stage 1 - **Rules**: Fast, cheap, catches obvious cases
```python
def rule_check(content):
    # Blocklist matching
    # Pattern detection (phone numbers, emails)
    # Length/character checks
```

Stage 2 - **ML classifier**: Trained on labeled examples
- Fast inference
- Categories: hate, violence, spam, adult, etc.
- Confidence scores for routing

Stage 3 - **LLM**: Complex/nuanced cases
- Context understanding
- Sarcasm, implied meaning
- More expensive, use selectively

**2. Decision routing**:
```python
def route_content(rule_result, ml_result, llm_result):
    if rule_result.blocked:
        return Action.BLOCK
    
    if ml_result.confidence > 0.95 and ml_result.safe:
        return Action.ALLOW
    
    if ml_result.confidence > 0.95 and not ml_result.safe:
        return Action.BLOCK
    
    # Uncertain: use LLM or human
    if llm_result:
        return Action.BLOCK if llm_result.harmful else Action.ALLOW
    
    return Action.QUEUE_FOR_REVIEW
```

**3. Human review interface**:
- Queue of uncertain cases
- Reviewer makes decision
- Decisions used to improve ML model

### Design decisions

**Why multi-stage?**
- Rules: Fast, predictable, no API cost
- ML: Good accuracy, scalable
- LLM: Best understanding, expensive
- Each stage reduces load on the next

**False positive vs false negative tradeoff**:
- Conservative: More human review, better user safety
- Liberal: Less review burden, some harmful content slips through
- Tune thresholds based on risk tolerance

**Feedback loop**:
- Human decisions improve ML model
- Track appeal/override rates
- Monitor for emerging abuse patterns

### Scaling considerations

- Async processing for non-real-time content
- Batch ML inference for efficiency
- Priority queues for high-risk content

---

## Design a recommendation system using LLMs

**The scenario**: Product recommendations that understand natural language preferences and can explain why items are recommended.

### Requirements clarification

- Catalog size: Thousands vs millions of items?
- Cold start: How to handle new users?
- Explainability: Need to explain recommendations?
- Latency: Real-time vs pre-computed?

### High-level architecture

```
User Profile + Query
        ↓
[Candidate Generation: narrow from 1M to 1000]
        ↓
[Ranking: narrow from 1000 to 50]
        ↓
[LLM Re-ranking + Explanation: top 10 with reasons]
        ↓
Personalized Recommendations + Explanations
```

### Key components

**1. Item embeddings**:
- Embed product descriptions, reviews, attributes
- Store in vector database
- Enables semantic similarity search

**2. User understanding**:
```python
def build_user_profile(user):
    return {
        "purchase_history": get_recent_purchases(user),
        "browsing_history": get_browsing_categories(user),
        "explicit_preferences": get_stated_preferences(user),
        "demographics": get_demographics(user),
    }
```

**3. Candidate generation**:
- Collaborative filtering: "users like you bought"
- Content-based: similar to past purchases
- Semantic search: match user preferences to item descriptions

**4. LLM ranking and explanation**:
```python
prompt = f"""
User preferences: {user_profile}
Recent query: {query}

Candidate products:
{candidates}

Rank these products for this user and explain why each is a good fit.
Return top 5 with explanations.
"""
```

### Design decisions

**Why not LLM for everything?**
- Catalog too large for LLM context
- Latency: LLM too slow for full catalog scan
- Cost: Can't call LLM for every candidate

**Hybrid approach**:
- Traditional ML for scale (candidate gen)
- LLM for intelligence (ranking, explanation)

**Explanation generation**:
- Users trust recommendations more when explained
- LLM naturally generates coherent explanations
- Can personalize explanation style

### Cold start handling

- New users: Popular items + demographic-based
- New items: Content-based on description
- Ask preferences explicitly on signup
