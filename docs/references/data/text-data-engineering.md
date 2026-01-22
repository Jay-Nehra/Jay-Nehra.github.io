# Text Data Engineering for LLM Applications

Text data is uniquely messy. It arrives in inconsistent formats, with encoding issues, structural variations, and quality problems that only reveal themselves downstream. This document explains how to engineer text data for LLM applications — not just cleaning scripts, but a systematic approach to making data usable.

We'll use polars as our execution tool. Not because it's fashionable, but because its execution model aligns with text data engineering at scale: lazy evaluation, expression-based transformations, and memory efficiency.

---

## 1. The Text Data Engineering Problem

Before we touch code, we need to understand what makes text data different and what "usable" actually means.

### Why Text Data Is Uniquely Messy

Numerical data has obvious validation: is it a number? Is it in range? Text data has no such clarity.

**Encoding chaos:** The same character can be represented multiple ways. A file might claim to be UTF-8 but contain Windows-1252 sequences. Mojibake — garbled text from encoding mismatches — looks like valid text until you read it.

**Structural inconsistency:** One CSV row has a clean paragraph. The next has HTML fragments. The third has escaped newlines. The fourth has the entire document in one cell. Same column, different universes.

**Invisible problems:** Whitespace variations, zero-width characters, homoglyphs (characters that look identical but aren't), and normalization differences all pass visual inspection but break downstream processing.

**Semantic ambiguity:** "N/A", "null", "none", "-", "" — all might mean missing data. Or they might be literal text. The source doesn't tell you.

### The Pipeline: Source → Clean → Validate → Transform → Artifacts

Text data engineering is a pipeline with distinct stages:

```
┌──────────┐    ┌─────────┐    ┌──────────┐    ┌───────────┐    ┌───────────┐
│  Source  │───▶│  Clean  │───▶│ Validate │───▶│ Transform │───▶│ Artifacts │
│ CSV/Excel│    │ Normalize│    │  Quality │    │  Reshape  │    │ Embeddings│
│  Files   │    │  Encode  │    │  Check   │    │   Chunk   │    │   JSONL   │
└──────────┘    └─────────┘    └──────────┘    └───────────┘    └───────────┘
```

Each stage has different concerns:

**Source:** Get data in without silent loss. Handle encoding, malformed rows, type inference failures.

**Clean:** Normalize to a consistent representation. Fix encoding issues. Remove structural noise.

**Validate:** Check quality. Flag problems. Decide what to keep, quarantine, or discard.

**Transform:** Reshape for downstream use. Chunk for embeddings. Format for fine-tuning.

**Artifacts:** Produce outputs that downstream systems expect. Embeddings, JSONL, metadata.

### What "Usable" Means for Each Use Case

"Usable" is not absolute. It depends on where the data goes:

**For embeddings/RAG:**
- Text must be chunked appropriately (not too long, not too short)
- Metadata must be preserved for filtering
- Duplicates must be removed (or you waste storage and confuse retrieval)
- Text must be clean enough that embeddings are meaningful

**For fine-tuning:**
- Instruction/response pairs must be correctly structured
- No data leakage between train/test splits
- Format must match what the training framework expects (usually JSONL)
- Quality must be high — garbage in training means garbage out

**For evaluation:**
- Gold-standard labels must be accurate
- Distribution should match production use
- Must be held out from any training data
- Reproducibility is essential

### Why Pandas Patterns Don't Scale for Text

Pandas is designed for tabular data with modest row counts. Text data breaks its assumptions:

**Memory model:** Pandas loads everything into memory as Python objects. A million documents with average 1KB each is 1GB of text, but pandas memory overhead makes it 5-10GB.

**Row operations:** Pandas `.apply()` iterates in Python. For text processing, this means Python string operations on every row — slow and memory-intensive.

**Copy semantics:** Many pandas operations copy data. Text is large. Copies are expensive.

Polars addresses these:
- Arrow memory format (columnar, zero-copy where possible)
- Expression-based operations (vectorized, no Python iteration)
- Lazy evaluation (query optimization before execution)

This isn't about polars being "better." It's about using the right tool for the data shape.

---

## 2. Polars for Text Data: Execution Model

Before processing text, understand how polars executes operations. This determines what's fast and what's slow.

### Expressions vs Row Operations

The fundamental shift from pandas to polars is from row operations to expressions.

**Pandas pattern (row-oriented):**
```python
# This iterates in Python — slow
df['clean'] = df['text'].apply(lambda x: x.strip().lower())
```

**Polars pattern (expression-oriented):**
```python
# This compiles to optimized operations — fast
df = df.with_columns(
    pl.col("text").str.strip_chars().str.to_lowercase().alias("clean")
)
```

The polars expression doesn't iterate. It describes a transformation that polars optimizes and executes in Rust.

This matters because text operations are numerous. If you have a million rows and 10 text transformations, the pandas approach runs 10 million Python function calls. The polars approach runs 10 vectorized operations.

### The Expression API for Text: `str` Namespace

Polars provides a `str` namespace with text operations:

```python
# Common text operations
pl.col("text").str.strip_chars()           # Trim whitespace
pl.col("text").str.to_lowercase()          # Lowercase
pl.col("text").str.replace_all(r"\s+", " ") # Normalize whitespace
pl.col("text").str.len_chars()             # Character count
pl.col("text").str.len_bytes()             # Byte count (for encoding issues)
pl.col("text").str.contains(r"pattern")    # Regex match
pl.col("text").str.extract(r"(pattern)")   # Regex capture
pl.col("text").str.split(" ")              # Split to list
```

These chain naturally:
```python
clean_expr = (
    pl.col("text")
    .str.strip_chars()
    .str.replace_all(r"\s+", " ")
    .str.to_lowercase()
    .alias("clean_text")
)
```

### Lazy vs Eager: When Each Matters

Polars has two modes:

**Eager (`pl.DataFrame`):** Operations execute immediately. Good for exploration.

**Lazy (`pl.LazyFrame`):** Operations build a query plan. Execution happens when you call `.collect()`. Good for pipelines.

For text data engineering, lazy evaluation is usually better:

```python
# Lazy: build the whole pipeline, optimize, then execute once
(
    pl.scan_csv("data.csv")  # Returns LazyFrame
    .with_columns(clean_text_expressions)
    .filter(quality_filter)
    .select(output_columns)
    .collect()  # Execute the optimized plan
)
```

Benefits:
- **Predicate pushdown:** Filters move early, reducing data processed
- **Projection pushdown:** Only needed columns are read
- **Common subexpression elimination:** Repeated calculations computed once
- **Memory efficiency:** Intermediate results don't materialize

For exploration, use eager:
```python
# Eager: see results immediately
df = pl.read_csv("sample.csv")  # Returns DataFrame
df.head()  # Immediate result
```

### What Actually Happens During Execution

When you call `.collect()` on a lazy frame:

1. **Plan optimization:** Polars rewrites your query for efficiency
2. **Chunk allocation:** Data is processed in chunks that fit in cache
3. **Parallel execution:** Operations parallelize across CPU cores
4. **Streaming (when enabled):** Data can stream through without full materialization

For text data, this means:
- Large files don't require proportionally large memory
- Text operations parallelize automatically
- The query planner may reorder operations for efficiency

This is why polars can handle text datasets that crash pandas.

---

## 3. Data Loading and Initial Profiling

The first step is getting data in correctly. This is where many pipelines silently corrupt data.

### Loading Messy CSVs

CSV is not a standard. Different tools produce different variants. Polars handles common issues:

```python
import polars as pl

# Basic load — often insufficient for messy data
df = pl.read_csv("data.csv")

# Robust load with common fixes
df = pl.read_csv(
    "data.csv",
    encoding="utf8-lossy",        # Replace invalid UTF-8 instead of failing
    ignore_errors=True,           # Skip malformed rows (logs count)
    truncate_ragged_lines=True,   # Handle rows with wrong column count
    quote_char='"',               # Explicit quote handling
    null_values=["", "NULL", "N/A", "null", "None"],  # Normalize nulls
    infer_schema_length=10000,    # More rows for type inference
)
```

**Encoding handling:**
- `encoding="utf8"` fails on invalid bytes (strict, good for clean data)
- `encoding="utf8-lossy"` replaces invalid bytes with replacement character
- For other encodings: `encoding="latin1"`, `encoding="cp1252"`, etc.

**The problem:** You often don't know the encoding. Detect it first:

```python
import chardet

def detect_encoding(filepath: str, sample_size: int = 100000) -> str:
    with open(filepath, 'rb') as f:
        result = chardet.detect(f.read(sample_size))
    return result['encoding']

encoding = detect_encoding("data.csv")
df = pl.read_csv("data.csv", encoding=encoding)
```

### Loading Excel Files

Excel files have their own problems:

```python
# Basic Excel load
df = pl.read_excel("data.xlsx")

# With sheet selection
df = pl.read_excel("data.xlsx", sheet_name="Sheet2")

# Reading all sheets
sheets = pl.read_excel("data.xlsx", sheet_id=0)  # Returns dict of DataFrames
```

Excel-specific issues:
- **Merged cells:** Polars unmerges, filling with nulls. You may need to forward-fill.
- **Type inference:** Excel stores everything as variants. Polars infers, sometimes wrongly.
- **Hidden rows/columns:** These are read by default.
- **Formulas:** Results are read, not formulas.

```python
# Handle forward-fill for unmerged cells
df = df.with_columns(
    pl.col("category").forward_fill()
)
```

### Initial Profiling

Before cleaning, understand what you have:

```python
def profile_text_dataframe(df: pl.DataFrame, text_col: str) -> dict:
    """Profile a text column for common issues."""
    text = pl.col(text_col)
    
    return df.select(
        # Basic stats
        pl.len().alias("row_count"),
        text.null_count().alias("null_count"),
        text.n_unique().alias("unique_count"),
        
        # Length distribution
        text.str.len_chars().mean().alias("mean_length"),
        text.str.len_chars().min().alias("min_length"),
        text.str.len_chars().max().alias("max_length"),
        text.str.len_chars().quantile(0.5).alias("median_length"),
        
        # Empty strings (different from null)
        (text == "").sum().alias("empty_string_count"),
        
        # Potential encoding issues (byte length != char length * expected)
        (text.str.len_bytes() != text.str.len_chars()).sum().alias("non_ascii_count"),
    ).to_dicts()[0]

# Usage
profile = profile_text_dataframe(df, "content")
print(f"Rows: {profile['row_count']}")
print(f"Nulls: {profile['null_count']} ({profile['null_count']/profile['row_count']*100:.1f}%)")
print(f"Mean length: {profile['mean_length']:.0f} chars")
```

### Detecting Encoding Issues

Mojibake — garbled text from encoding mismatches — has patterns:

```python
# Common mojibake patterns (UTF-8 interpreted as Latin-1)
MOJIBAKE_PATTERNS = [
    r"â€™",  # Right single quote
    r"â€œ",  # Left double quote
    r"â€",   # Right double quote
    r"Ã©",   # é
    r"Ã¨",   # è
    r"Ã ",   # à
]

def detect_mojibake(df: pl.DataFrame, text_col: str) -> pl.DataFrame:
    """Flag rows with likely encoding issues."""
    patterns = "|".join(MOJIBAKE_PATTERNS)
    return df.with_columns(
        pl.col(text_col).str.contains(patterns).alias("has_mojibake")
    )

# Check how many rows have issues
mojibake_count = df.filter(pl.col("has_mojibake")).height
print(f"Rows with encoding issues: {mojibake_count}")
```

---

## 4. Text Cleaning Patterns

Cleaning is not a single operation. It's a hierarchy of transformations, each building on the last.

### The Normalization Hierarchy

Apply cleaning in order, from most fundamental to most domain-specific:

1. **Encoding:** Fix byte-level issues first
2. **Unicode normalization:** Canonical forms (NFC/NFD)
3. **Whitespace:** Normalize spaces, tabs, newlines
4. **Case:** Lowercase (if appropriate)
5. **Punctuation:** Normalize quotes, dashes, etc.
6. **Domain-specific:** Remove boilerplate, extract content

Each level depends on previous levels being clean.

### Encoding Fixes

If you detected mojibake, fix it:

```python
def fix_common_mojibake(text: str) -> str:
    """Fix common UTF-8 → Latin-1 mojibake."""
    replacements = {
        "â€™": "'",
        "â€œ": '"',
        "â€": '"',
        "â€"": "—",
        "â€"": "–",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã¢": "â",
        "Ã®": "î",
        "Ã´": "ô",
        "Ã»": "û",
        "Ã§": "ç",
    }
    for wrong, right in replacements.items():
        text = text.replace(wrong, right)
    return text

# In polars, use replace_all for each pattern
# Or use map_elements for complex fixes (slower but flexible)
df = df.with_columns(
    pl.col("text").map_elements(fix_common_mojibake, return_dtype=pl.Utf8).alias("text_fixed")
)
```

For systematic encoding repair, consider the `ftfy` library:

```python
import ftfy

df = df.with_columns(
    pl.col("text").map_elements(ftfy.fix_text, return_dtype=pl.Utf8).alias("text_fixed")
)
```

Note: `map_elements` is slow (Python iteration). Use it for complex fixes, not simple transformations.

### Unicode Normalization

The same visual character can have multiple Unicode representations:
- "é" can be U+00E9 (precomposed) or U+0065 U+0301 (e + combining accent)

Normalize to a consistent form:

```python
import unicodedata

def normalize_unicode(text: str) -> str:
    return unicodedata.normalize("NFC", text)

df = df.with_columns(
    pl.col("text").map_elements(normalize_unicode, return_dtype=pl.Utf8).alias("text_normalized")
)
```

NFC (Composed) is usually best for text processing — it produces the shortest representation.

### Whitespace Normalization

Whitespace is surprisingly complex:
- Regular space (U+0020)
- Non-breaking space (U+00A0)
- Various Unicode spaces (em space, en space, thin space...)
- Tabs, newlines, carriage returns
- Zero-width characters

```python
# Normalize all whitespace to single spaces
def normalize_whitespace(df: pl.DataFrame, col: str) -> pl.DataFrame:
    return df.with_columns(
        pl.col(col)
        .str.replace_all(r"[\t\r\n]+", " ")  # Convert tabs/newlines to spaces
        .str.replace_all(r"\s+", " ")         # Collapse multiple spaces
        .str.strip_chars()                     # Trim edges
        .alias(col)
    )
```

For aggressive cleaning (remove all non-standard whitespace):

```python
# Replace all Unicode whitespace with regular space
df = df.with_columns(
    pl.col("text")
    .str.replace_all(r"[\u00A0\u1680\u2000-\u200A\u202F\u205F\u3000]", " ")
    .str.replace_all(r"\s+", " ")
    .str.strip_chars()
    .alias("text_clean")
)
```

### Structural Cleaning

Text often contains structural elements that aren't content:

**HTML/XML:**
```python
import re

def strip_html(text: str) -> str:
    """Remove HTML tags, keeping text content."""
    # Remove tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode entities
    import html
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df = df.with_columns(
    pl.col("text").map_elements(strip_html, return_dtype=pl.Utf8).alias("text_clean")
)
```

For robust HTML handling, use `BeautifulSoup`:

```python
from bs4 import BeautifulSoup

def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, 'html.parser')
    # Remove script and style elements
    for script in soup(["script", "style"]):
        script.decompose()
    return soup.get_text(separator=' ', strip=True)
```

### The "Cleaning as Transformation" Mindset

Don't overwrite original data. Add cleaned columns:

```python
df = (
    df
    .with_columns(
        # Keep original
        pl.col("raw_text").alias("text_original"),
        
        # Add cleaning stages
        pl.col("raw_text")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .alias("text_normalized"),
    )
    .with_columns(
        pl.col("text_normalized")
        .str.to_lowercase()
        .alias("text_lowercase"),
    )
)
```

This enables:
- Debugging (compare stages)
- Rollback (original is preserved)
- Different downstream uses (some need lowercase, some don't)

---

## 5. Data Quality and Validation

Cleaning is not enough. You need to verify quality systematically.

### Quality Dimensions for Text

**Completeness:** Is the data present? Nulls, empty strings, truncation.

**Consistency:** Is the format uniform? Mixed encodings, varying structures.

**Accuracy:** Is the content correct? Mojibake, corruption, wrong data in columns.

**Timeliness:** Is the data current? Outdated content, stale snapshots.

**Uniqueness:** Is each record distinct? Duplicates, near-duplicates.

### Validation Expressions

Build reusable validation expressions:

```python
def create_text_validators(col: str, min_length: int = 10, max_length: int = 50000):
    """Create validation expressions for a text column."""
    text = pl.col(col)
    
    return {
        "is_null": text.is_null(),
        "is_empty": text == "",
        "too_short": text.str.len_chars() < min_length,
        "too_long": text.str.len_chars() > max_length,
        "has_mojibake": text.str.contains(r"â€|Ã©|Ã¨"),
        "mostly_punctuation": (
            text.str.count_matches(r"[^\w\s]") / text.str.len_chars() > 0.5
        ),
        "mostly_numbers": (
            text.str.count_matches(r"\d") / text.str.len_chars() > 0.8
        ),
    }

# Apply validators
validators = create_text_validators("content")

df = df.with_columns(
    validators["is_null"].alias("flag_null"),
    validators["too_short"].alias("flag_short"),
    validators["has_mojibake"].alias("flag_encoding"),
)

# Count issues
issues = df.select(
    pl.col("flag_null").sum().alias("null_count"),
    pl.col("flag_short").sum().alias("short_count"),
    pl.col("flag_encoding").sum().alias("encoding_count"),
).to_dicts()[0]

print(f"Issues found: {issues}")
```

### Text Distribution Profiling

Understand your data's statistical properties:

```python
def profile_text_distribution(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Generate distribution statistics for text column."""
    text = pl.col(col)
    
    return df.select(
        # Length distribution
        text.str.len_chars().min().alias("length_min"),
        text.str.len_chars().max().alias("length_max"),
        text.str.len_chars().mean().alias("length_mean"),
        text.str.len_chars().std().alias("length_std"),
        text.str.len_chars().quantile(0.25).alias("length_p25"),
        text.str.len_chars().quantile(0.50).alias("length_p50"),
        text.str.len_chars().quantile(0.75).alias("length_p75"),
        text.str.len_chars().quantile(0.95).alias("length_p95"),
        
        # Word count (approximate)
        text.str.count_matches(r"\S+").mean().alias("word_count_mean"),
        
        # Sentence count (approximate)
        text.str.count_matches(r"[.!?]+").mean().alias("sentence_count_mean"),
    )

profile = profile_text_distribution(df, "content")
print(profile)
```

### Flagging vs Filtering vs Fixing

Not all issues require the same response:

**Flag (keep but mark):**
- Unusual but potentially valid data
- Issues you want to investigate later
- Data needed for completeness even if imperfect

**Quarantine (separate):**
- Data that might be salvageable
- Issues that need manual review
- High-value data with quality problems

**Filter (remove):**
- Clear garbage
- Irrecoverable corruption
- Duplicates

**Fix (transform):**
- Systematic, correctable issues
- Encoding problems with known fixes
- Normalization differences

```python
# Example: comprehensive quality handling
df_processed = (
    df
    # Add quality flags
    .with_columns(
        (pl.col("content").is_null() | (pl.col("content") == "")).alias("is_empty"),
        (pl.col("content").str.len_chars() < 50).alias("is_too_short"),
        pl.col("content").str.contains(r"â€").alias("has_encoding_issue"),
    )
    # Create quality tier
    .with_columns(
        pl.when(pl.col("is_empty"))
        .then(pl.lit("discard"))
        .when(pl.col("has_encoding_issue"))
        .then(pl.lit("quarantine"))
        .when(pl.col("is_too_short"))
        .then(pl.lit("review"))
        .otherwise(pl.lit("good"))
        .alias("quality_tier")
    )
)

# Split by tier
good_data = df_processed.filter(pl.col("quality_tier") == "good")
quarantine_data = df_processed.filter(pl.col("quality_tier") == "quarantine")
review_data = df_processed.filter(pl.col("quality_tier") == "review")

print(f"Good: {good_data.height}, Quarantine: {quarantine_data.height}, Review: {review_data.height}")
```

---

## 6. Preparing for Embeddings and RAG

The most common destination for processed text is embedding pipelines. This requires specific transformations.

### Chunking in Polars

Text must be chunked before embedding. Polars can handle this:

**Simple character-based chunking:**

```python
def chunk_text_simple(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# Apply to dataframe
df = df.with_columns(
    pl.col("content")
    .map_elements(lambda x: chunk_text_simple(x, 500, 50), return_dtype=pl.List(pl.Utf8))
    .alias("chunks")
)

# Explode to one row per chunk
df_chunks = df.explode("chunks")
```

**Sentence-aware chunking:**

```python
import re

def chunk_by_sentences(text: str, max_chars: int = 500) -> list:
    """Chunk text at sentence boundaries."""
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if current_length + len(sentence) > max_chars and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
            current_length = 0
        current_chunk.append(sentence)
        current_length += len(sentence) + 1
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks
```

### Metadata Preservation

When you chunk, preserve source metadata:

```python
# Before chunking
df = pl.DataFrame({
    "doc_id": ["doc1", "doc2"],
    "source": ["report.pdf", "manual.pdf"],
    "content": ["Long text here...", "Another long text..."],
})

# After chunking with metadata preserved
df_chunks = (
    df
    .with_columns(
        pl.col("content")
        .map_elements(chunk_by_sentences, return_dtype=pl.List(pl.Utf8))
        .alias("chunks")
    )
    .with_row_index("doc_index")  # Add original row index
    .explode("chunks")
    .with_columns(
        # Add chunk index within document
        pl.col("chunks").cum_count().over("doc_id").alias("chunk_index")
    )
    .rename({"chunks": "chunk_text"})
)

# Result has: doc_id, source, chunk_text, doc_index, chunk_index
```

### Deduplication

Duplicate content wastes storage and confuses retrieval.

**Exact deduplication:**

```python
# Remove exact duplicates
df_unique = df.unique(subset=["content"])
print(f"Removed {df.height - df_unique.height} exact duplicates")
```

**Near-exact deduplication (normalize first):**

```python
# Normalize then dedupe
df_unique = (
    df
    .with_columns(
        pl.col("content")
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .str.strip_chars()
        .alias("content_normalized")
    )
    .unique(subset=["content_normalized"])
    .drop("content_normalized")
)
```

**Fuzzy deduplication (MinHash):**

For near-duplicate detection at scale, use MinHash:

```python
from datasketch import MinHash, MinHashLSH

def create_minhash(text: str, num_perm: int = 128) -> MinHash:
    """Create MinHash signature for text."""
    m = MinHash(num_perm=num_perm)
    for word in text.lower().split():
        m.update(word.encode('utf8'))
    return m

# Build LSH index
lsh = MinHashLSH(threshold=0.8, num_perm=128)
minhashes = {}

for idx, row in enumerate(df.iter_rows(named=True)):
    mh = create_minhash(row["content"])
    minhashes[idx] = mh
    lsh.insert(idx, mh)

# Find duplicates
duplicate_groups = []
seen = set()

for idx, mh in minhashes.items():
    if idx in seen:
        continue
    result = lsh.query(mh)
    if len(result) > 1:
        duplicate_groups.append(result)
        seen.update(result)

# Keep first from each group, remove rest
indices_to_remove = set()
for group in duplicate_groups:
    indices_to_remove.update(list(group)[1:])

df_deduped = df.with_row_index().filter(~pl.col("index").is_in(indices_to_remove))
```

### Output Formats for Vector Databases

Vector databases expect specific formats:

**For Pinecone:**

```python
def prepare_for_pinecone(df: pl.DataFrame) -> list:
    """Convert dataframe to Pinecone upsert format."""
    records = []
    for row in df.iter_rows(named=True):
        records.append({
            "id": row["chunk_id"],
            "values": row["embedding"],  # You'll compute this separately
            "metadata": {
                "source": row["source"],
                "doc_id": row["doc_id"],
                "chunk_index": row["chunk_index"],
                "text": row["chunk_text"][:1000],  # Pinecone metadata limit
            }
        })
    return records
```

**For ChromaDB:**

```python
def prepare_for_chroma(df: pl.DataFrame) -> dict:
    """Convert dataframe to ChromaDB add format."""
    return {
        "ids": df["chunk_id"].to_list(),
        "documents": df["chunk_text"].to_list(),
        "metadatas": [
            {"source": row["source"], "doc_id": row["doc_id"]}
            for row in df.iter_rows(named=True)
        ]
    }
```

---

## 7. Preparing Fine-Tuning Datasets

Fine-tuning requires specific formats and careful handling to avoid contamination.

### Instruction/Response Pair Creation

Fine-tuning datasets typically need instruction-response pairs:

```python
def create_instruction_pairs(
    df: pl.DataFrame,
    context_col: str,
    question_col: str,
    answer_col: str
) -> pl.DataFrame:
    """Create instruction-response pairs for fine-tuning."""
    return df.with_columns(
        pl.format(
            "Given the following context:\n\n{}\n\nAnswer this question: {}",
            pl.col(context_col),
            pl.col(question_col)
        ).alias("instruction"),
        pl.col(answer_col).alias("response")
    ).select(["instruction", "response"])
```

**Chat format for instruction tuning:**

```python
def format_as_chat(df: pl.DataFrame) -> pl.DataFrame:
    """Format as chat messages for instruction tuning."""
    return df.with_columns(
        pl.struct([
            pl.lit("system").alias("role"),
            pl.lit("You are a helpful assistant.").alias("content")
        ]).alias("system_message"),
        pl.struct([
            pl.lit("user").alias("role"),
            pl.col("instruction").alias("content")
        ]).alias("user_message"),
        pl.struct([
            pl.lit("assistant").alias("role"),
            pl.col("response").alias("content")
        ]).alias("assistant_message")
    ).with_columns(
        pl.concat_list([
            pl.col("system_message"),
            pl.col("user_message"),
            pl.col("assistant_message")
        ]).alias("messages")
    ).select(["messages"])
```

### JSONL Output

Most training frameworks expect JSONL:

```python
import json

def export_to_jsonl(df: pl.DataFrame, path: str, message_col: str = "messages"):
    """Export dataframe to JSONL format for training."""
    with open(path, 'w') as f:
        for row in df.iter_rows(named=True):
            record = {"messages": row[message_col]}
            f.write(json.dumps(record) + '\n')

# Verify format
def validate_jsonl(path: str) -> dict:
    """Validate JSONL file format."""
    issues = []
    line_count = 0
    
    with open(path) as f:
        for i, line in enumerate(f):
            line_count += 1
            try:
                record = json.loads(line)
                if "messages" not in record:
                    issues.append(f"Line {i}: missing 'messages' field")
            except json.JSONDecodeError as e:
                issues.append(f"Line {i}: invalid JSON - {e}")
    
    return {"line_count": line_count, "issues": issues}
```

### Train/Validation/Test Splitting

Proper splits are essential:

```python
def stratified_split(
    df: pl.DataFrame,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_col: str = None,
    seed: int = 42
) -> tuple:
    """Split dataframe into train/val/test sets."""
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001
    
    # Shuffle
    df = df.sample(fraction=1.0, seed=seed, shuffle=True)
    
    n = df.height
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_df = df.slice(0, train_end)
    val_df = df.slice(train_end, val_end - train_end)
    test_df = df.slice(val_end, n - val_end)
    
    return train_df, val_df, test_df

train, val, test = stratified_split(df)
print(f"Train: {train.height}, Val: {val.height}, Test: {test.height}")
```

### Decontamination

Test data must not appear in training data:

```python
def decontaminate(
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    text_col: str
) -> pl.DataFrame:
    """Remove any test examples that appear in training set."""
    # Create normalized versions for comparison
    train_normalized = (
        train_df
        .with_columns(
            pl.col(text_col)
            .str.to_lowercase()
            .str.replace_all(r"\s+", " ")
            .alias("__normalized")
        )
        .select("__normalized")
        .unique()
    )
    
    test_normalized = test_df.with_columns(
        pl.col(text_col)
        .str.to_lowercase()
        .str.replace_all(r"\s+", " ")
        .alias("__normalized")
    )
    
    # Find contaminated rows
    contaminated = test_normalized.join(
        train_normalized,
        on="__normalized",
        how="inner"
    )
    
    print(f"Found {contaminated.height} contaminated test examples")
    
    # Return clean test set
    clean_test = test_normalized.join(
        train_normalized,
        on="__normalized",
        how="anti"
    ).drop("__normalized")
    
    return clean_test
```

---

## 8. Preparing Evaluation Datasets

Evaluation datasets require special care — they're the ground truth for measuring system quality.

### Gold-Standard Annotation

Evaluation data needs human-verified labels:

```python
def create_annotation_template(df: pl.DataFrame, output_path: str):
    """Create a template for human annotation."""
    # Select columns for annotation
    template = df.select([
        "id",
        "content",
        pl.lit("").alias("label"),  # To be filled by annotator
        pl.lit("").alias("notes"),  # Annotator notes
    ])
    
    # Export for annotation (CSV is often easiest for non-technical annotators)
    template.write_csv(output_path)
    print(f"Annotation template written to {output_path}")

def load_annotations(
    original_df: pl.DataFrame,
    annotations_path: str
) -> pl.DataFrame:
    """Load and merge annotations back to original data."""
    annotations = pl.read_csv(annotations_path)
    
    # Validate
    missing = original_df.join(
        annotations.select("id"),
        on="id",
        how="anti"
    )
    if missing.height > 0:
        print(f"Warning: {missing.height} items not annotated")
    
    # Merge
    return original_df.join(annotations.select(["id", "label", "notes"]), on="id")
```

### Inter-Annotator Agreement

When multiple annotators label the same data, measure agreement:

```python
def calculate_agreement(annotations_df: pl.DataFrame) -> dict:
    """Calculate inter-annotator agreement metrics."""
    # Assuming columns: id, annotator, label
    
    # Get items annotated by multiple annotators
    multi_annotated = (
        annotations_df
        .group_by("id")
        .agg(pl.count().alias("annotator_count"))
        .filter(pl.col("annotator_count") > 1)
    )
    
    # Calculate exact agreement
    agreements = (
        annotations_df
        .join(multi_annotated.select("id"), on="id")
        .group_by("id")
        .agg(
            pl.col("label").n_unique().alias("unique_labels")
        )
        .with_columns(
            (pl.col("unique_labels") == 1).alias("agreed")
        )
    )
    
    agreement_rate = agreements["agreed"].mean()
    
    return {
        "items_with_multiple_annotations": multi_annotated.height,
        "exact_agreement_rate": agreement_rate,
    }
```

### Sampling Strategies

Evaluation sets should represent production distribution:

```python
def stratified_sample(
    df: pl.DataFrame,
    n_samples: int,
    stratify_col: str,
    seed: int = 42
) -> pl.DataFrame:
    """Sample maintaining category distribution."""
    # Calculate samples per category
    category_counts = df.group_by(stratify_col).count()
    total = df.height
    
    samples = []
    for row in category_counts.iter_rows(named=True):
        category = row[stratify_col]
        proportion = row["count"] / total
        n_category = max(1, int(n_samples * proportion))
        
        category_df = df.filter(pl.col(stratify_col) == category)
        sample = category_df.sample(n=min(n_category, category_df.height), seed=seed)
        samples.append(sample)
    
    return pl.concat(samples)
```

### Versioning and Reproducibility

Evaluation datasets must be versioned:

```python
import hashlib
from datetime import datetime

def version_dataset(df: pl.DataFrame, name: str, output_dir: str) -> dict:
    """Version a dataset with hash and metadata."""
    # Compute content hash
    content_str = df.write_csv()
    content_hash = hashlib.sha256(content_str.encode()).hexdigest()[:12]
    
    # Create versioned filename
    timestamp = datetime.now().strftime("%Y%m%d")
    filename = f"{name}_{timestamp}_{content_hash}.parquet"
    filepath = f"{output_dir}/{filename}"
    
    # Save
    df.write_parquet(filepath)
    
    # Return metadata
    return {
        "name": name,
        "version": f"{timestamp}_{content_hash}",
        "filepath": filepath,
        "row_count": df.height,
        "columns": df.columns,
        "created_at": datetime.now().isoformat(),
    }
```

---

## 9. Pipeline Patterns

Individual operations combine into pipelines. Managing pipelines well ensures reproducibility and maintainability.

### Reproducible Pipelines

Pipelines should produce identical outputs given identical inputs:

```python
from dataclasses import dataclass
from typing import Optional
import json

@dataclass
class PipelineConfig:
    """Configuration for text processing pipeline."""
    input_path: str
    output_dir: str
    
    # Cleaning options
    normalize_unicode: bool = True
    lowercase: bool = True
    min_length: int = 50
    max_length: int = 50000
    
    # Chunking options
    chunk_size: int = 500
    chunk_overlap: int = 50
    
    # Deduplication
    deduplicate: bool = True
    dedupe_threshold: float = 0.9
    
    # Random seed for reproducibility
    seed: int = 42
    
    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> "PipelineConfig":
        return cls(**json.loads(json_str))


def run_pipeline(config: PipelineConfig) -> pl.DataFrame:
    """Run the complete text processing pipeline."""
    
    # Load
    df = pl.read_csv(config.input_path)
    
    # Clean
    df = df.with_columns(
        pl.col("content")
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        .alias("content_clean")
    )
    
    if config.lowercase:
        df = df.with_columns(
            pl.col("content_clean").str.to_lowercase()
        )
    
    # Filter by length
    df = df.filter(
        (pl.col("content_clean").str.len_chars() >= config.min_length) &
        (pl.col("content_clean").str.len_chars() <= config.max_length)
    )
    
    # Deduplicate
    if config.deduplicate:
        df = df.unique(subset=["content_clean"])
    
    # Chunk
    df = df.with_columns(
        pl.col("content_clean")
        .map_elements(
            lambda x: chunk_text_simple(x, config.chunk_size, config.chunk_overlap),
            return_dtype=pl.List(pl.Utf8)
        )
        .alias("chunks")
    ).explode("chunks")
    
    # Save config alongside output
    config_path = f"{config.output_dir}/pipeline_config.json"
    with open(config_path, 'w') as f:
        f.write(config.to_json())
    
    return df
```

### Checkpointing Intermediate Results

For long pipelines, save intermediate results:

```python
def run_with_checkpoints(config: PipelineConfig) -> pl.DataFrame:
    """Run pipeline with checkpoints for resumability."""
    
    checkpoint_dir = f"{config.output_dir}/checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Stage 1: Load and initial clean
    stage1_path = f"{checkpoint_dir}/stage1_loaded.parquet"
    if os.path.exists(stage1_path):
        print("Loading stage 1 from checkpoint")
        df = pl.read_parquet(stage1_path)
    else:
        print("Running stage 1: load and initial clean")
        df = pl.read_csv(config.input_path)
        df = initial_clean(df)
        df.write_parquet(stage1_path)
    
    # Stage 2: Quality filtering
    stage2_path = f"{checkpoint_dir}/stage2_filtered.parquet"
    if os.path.exists(stage2_path):
        print("Loading stage 2 from checkpoint")
        df = pl.read_parquet(stage2_path)
    else:
        print("Running stage 2: quality filtering")
        df = quality_filter(df, config)
        df.write_parquet(stage2_path)
    
    # Stage 3: Chunking
    stage3_path = f"{checkpoint_dir}/stage3_chunked.parquet"
    if os.path.exists(stage3_path):
        print("Loading stage 3 from checkpoint")
        df = pl.read_parquet(stage3_path)
    else:
        print("Running stage 3: chunking")
        df = chunk_documents(df, config)
        df.write_parquet(stage3_path)
    
    return df
```

### Incremental Processing

For growing datasets, process only new data:

```python
def process_incremental(
    new_data_path: str,
    existing_data_path: str,
    config: PipelineConfig
) -> pl.DataFrame:
    """Process only new data and merge with existing."""
    
    # Load existing
    existing = pl.read_parquet(existing_data_path)
    existing_ids = set(existing["doc_id"].to_list())
    
    # Load new
    new_data = pl.read_csv(new_data_path)
    
    # Filter to truly new
    truly_new = new_data.filter(~pl.col("doc_id").is_in(existing_ids))
    print(f"Processing {truly_new.height} new documents")
    
    if truly_new.height == 0:
        return existing
    
    # Process new data
    processed_new = run_pipeline_on_df(truly_new, config)
    
    # Merge
    combined = pl.concat([existing, processed_new])
    
    return combined
```

### Data Versioning Concepts

Track data lineage:

```python
@dataclass
class DatasetVersion:
    """Track dataset version and lineage."""
    name: str
    version: str
    created_at: str
    row_count: int
    content_hash: str
    
    # Lineage
    source_path: str
    pipeline_config_hash: str
    parent_version: Optional[str] = None
    
    def to_dict(self) -> dict:
        return self.__dict__

def create_version_manifest(
    df: pl.DataFrame,
    name: str,
    config: PipelineConfig,
    source_path: str,
    parent_version: str = None
) -> DatasetVersion:
    """Create version manifest for a processed dataset."""
    
    # Compute hashes
    content_hash = hashlib.sha256(
        df.write_csv().encode()
    ).hexdigest()[:12]
    
    config_hash = hashlib.sha256(
        config.to_json().encode()
    ).hexdigest()[:12]
    
    return DatasetVersion(
        name=name,
        version=f"v{datetime.now().strftime('%Y%m%d')}_{content_hash}",
        created_at=datetime.now().isoformat(),
        row_count=df.height,
        content_hash=content_hash,
        source_path=source_path,
        pipeline_config_hash=config_hash,
        parent_version=parent_version,
    )
```

---

## Summary: The Text Data Engineering Mental Model

Text data engineering is a systematic pipeline from messy sources to usable artifacts.

**Loading** is where most problems start. Detect encoding, handle malformed data, profile before cleaning.

**Cleaning** is hierarchical. Fix encoding first, then normalize, then domain-specific cleaning. Preserve originals.

**Validation** is systematic. Build reusable validators. Flag, quarantine, or filter based on severity.

**Transformation** depends on destination. Chunking for embeddings, formatting for fine-tuning, sampling for evaluation.

**Pipelines** must be reproducible. Use configs, checkpoints, and versioning.

**Polars** is the right tool for text at scale. Expressions over row operations. Lazy evaluation for large datasets.

---

## Interview Framing

**"How would you prepare a large text dataset for RAG?"**

"First, load with explicit encoding handling — detect encoding before reading, use lossy mode if needed. Profile to understand the data: length distribution, null rate, encoding issues. Clean systematically: unicode normalization, whitespace normalization, structural cleaning like HTML stripping. Validate quality and quarantine problematic records rather than silently dropping them. Then chunk with appropriate overlap, deduplicate to avoid redundant embeddings, and preserve metadata for filtering. The key is making it reproducible — config files, checkpoints, and version manifests."

**"Why polars over pandas for text data?"**

"Text data is memory-intensive. Pandas loads everything as Python objects with significant overhead. Polars uses Arrow format which is more memory-efficient, and its expression-based operations avoid Python iteration. For a million documents, pandas might need 10GB; polars might need 2GB. The lazy evaluation also means I can build complex pipelines and polars optimizes the execution plan. For text specifically, the `str` namespace provides vectorized operations that would require slow `.apply()` calls in pandas."

**"What are the main pitfalls in text data preparation?"**

"Silent encoding corruption is the biggest — data looks valid but contains mojibake. Detect encoding explicitly and validate after loading. Chunking mistakes: splitting mid-sentence or mid-word breaks semantic meaning. Deduplication failures: near-duplicates waste embedding storage and confuse retrieval. Data leakage: test data appearing in training sets. And reproducibility: if you can't regenerate the same output from the same input, you can't debug problems or track improvements."
