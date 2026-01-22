---
tags:
  - python
  - serialization
  - stdlib
---

# Python's json Module

## Part 1: Architecture

### The Mental Model: Translation Between Worlds

Think of our Python program as one country, and the outside world (files, APIs, databases) as another. These countries speak different languages:

- **Python's language**: Objects in memory—dicts, lists, integers, custom classes, references, pointers
- **The outside world's language**: Text—sequences of characters that can be written to disk or sent over a network

**Serialization** is translation from Python → Text.  
**Deserialization** is translation from Text → Python.

JSON is one specific "lingua franca"—a text format that both sides agreed to use. It's not the only one (there's also XML, YAML, pickle, protobuf), but it won the popularity contest because:

1. Human-readable (we can open it in a text editor)
2. Language-agnostic (JavaScript, Go, Rust all speak it)
3. Simple (only 6 data types)

### What Problem Does This Solve?

**The fundamental problem**: Python objects live in RAM. RAM is volatile. When our program ends, everything disappears.

Without serialization, we cannot:
- Save program state to a file
- Send data to another process or machine
- Store data in a database
- Communicate with a web API

**The naive approach** would be to use Python's `str()`:

```python
data = {"user": "jay", "count": 42}
text = str(data)  # "{'user': 'jay', 'count': 42}"
```

This looks like it works, but it fails immediately:

```python
# Try to get it back
recovered = eval(text)  # Works... but DANGEROUS (code injection)
```

And it fails completely for other languages:

```python
# JavaScript cannot parse Python's repr format
# {'user': 'jay'} is not valid JSON (single quotes!)
```

JSON solves this by defining a strict, unambiguous text format that every language can parse.

### The Machinery: What Actually Happens

#### When you call `json.dumps(obj)`

This is not "converting to a string." It's a **recursive traversal** of your object graph with **type-based dispatch**.

Step by step:

1. **Entry**: `json.dumps({"user": "jay", "scores": [1, 2, 3]})`

2. **Type check**: What is this object?
   - It's a `dict` → JSON object, use `{}`
   
3. **Iterate keys**: For each key-value pair:
   - Key "user" → must be a string (JSON requirement)
   - Value "jay" → type check: it's a `str` → wrap in quotes
   - Key "scores" → string, good
   - Value `[1, 2, 3]` → type check: it's a `list` → **recurse**

4. **Recursion**: Now processing the list:
   - It's a `list` → JSON array, use `[]`
   - Element 1 → type check: `int` → write as number
   - Element 2 → type check: `int` → write as number
   - Element 3 → type check: `int` → write as number

5. **Assembly**: Build the final string: `{"user": "jay", "scores": [1, 2, 3]}`

**The critical insight**: At step 2, if the type check fails (e.g., you pass a `datetime` object), the encoder doesn't know what to do. It raises `TypeError: Object of type datetime is not JSON serializable`.

This is not a bug—it's the encoder saying "I don't have a translation rule for this type."

#### The Type Dispatch Table

The json module has a hardcoded mapping:

| Python Type | JSON Type | Notes |
|-------------|-----------|-------|
| `dict` | object `{}` | Keys MUST be strings |
| `list`, `tuple` | array `[]` | Tuples become lists (no tuple in JSON) |
| `str` | string `""` | Unicode handled automatically |
| `int`, `float` | number | `float('inf')` fails (not valid JSON) |
| `True`, `False` | `true`, `false` | Note: lowercase in JSON |
| `None` | `null` | |

**Everything else fails.** No datetime. No set. No custom classes. No bytes.

#### When you call `json.loads(text)`

The reverse process:

1. **Parsing**: Read characters, identify tokens (`{`, `"user"`, `:`, etc.)
2. **Validation**: Is this valid JSON syntax? If not, raise `json.JSONDecodeError`
3. **Construction**: Build Python objects based on what was parsed
   - `{}` → `dict`
   - `[]` → `list`
   - `"..."` → `str`
   - numbers → `int` or `float` (depends on decimal point)
   - `true`/`false` → `True`/`False`
   - `null` → `None`

**Important**: The parser has no memory of what the original Python types were. If you serialized a tuple, you get a list back. If you had a custom class, you get a dict back (if you even managed to serialize it).

### Key Concepts (Behavioral Definitions)

**Serialization**
- What we might assume: "Converting to string"
- What it actually means: Recursively traversing an object graph and applying type-specific encoding rules to produce a text representation
- Why this matters: Understanding this explains WHY certain types fail—there's no rule for them

**JSON-serializable**
- What we might assume: "Anything can be converted if we try hard enough"
- What it actually means: The object's type (and all nested types) must be in the encoder's dispatch table
- Why this matters: We need to either stick to basic types OR extend the encoder

**Encoder/Decoder**
- What we might assume: "Magic functions that do the conversion"
- What it actually means: The `JSONEncoder` class contains the type dispatch logic; we can subclass it to add rules for new types
- Why this matters: This is how we make custom objects serializable

**Round-trip**
- What we might assume: `loads(dumps(x)) == x` always
- What it actually means: Only true for JSON-native types. Tuples become lists. Custom objects need special handling to reconstruct.
- Why this matters: Don't assume we get the same thing back

### Design Decisions: Why Is It This Way?

**Why so few types?**

JSON was designed for JavaScript, which has fewer built-in types than Python. The designers prioritized:
- Interoperability (every language can implement these 6 types)
- Simplicity (easy to parse, hard to get wrong)

The alternative would be a Python-specific format (like `pickle`), which is more powerful but:
- Not readable by other languages
- Security risk (pickle can execute arbitrary code)

**Why do keys have to be strings?**

In JavaScript, object keys are always strings. JSON inherited this. If you try:

```python
json.dumps({1: "one", 2: "two"})  # Works! Keys become "1", "2"
json.dumps({(1, 2): "tuple key"})  # Fails! Can't stringify tuple
```

The module silently converts int keys to strings, but fails on anything more complex.

**Why no datetime?**

There's no universal agreement on how to represent dates in JSON. Options:
- ISO 8601 string: `"2024-01-15T10:30:00Z"`
- Unix timestamp: `1705315800`
- Separate fields: `{"year": 2024, "month": 1, "day": 15}`

Rather than pick one, the json module forces you to decide. This is intentional.

### What Breaks If You Misunderstand

**Mistake 1: Assuming anything is serializable**

```python
from datetime import datetime

data = {"created": datetime.now()}
json.dumps(data)  # TypeError!
```

**Mistake 2: Expecting round-trip fidelity**

```python
original = {"items": (1, 2, 3)}  # tuple
recovered = json.loads(json.dumps(original))
print(type(recovered["items"]))  # <class 'list'> — NOT tuple!
```

**Mistake 3: Ignoring encoding in file operations**

```python
# This can fail with non-ASCII characters on some systems
with open("data.json", "w") as f:  # No encoding specified!
    json.dump({"name": "Müller"}, f)

# Safe version:
with open("data.json", "w", encoding="utf-8") as f:
    json.dump({"name": "Müller"}, f)
```

**Mistake 4: Using `json.load()` on untrusted data without limits**

```python
# Malicious JSON can be deeply nested, causing stack overflow
# Or contain huge strings, exhausting memory
data = json.loads(untrusted_input)  # Dangerous!
```

---

## Part 2: Scenarios

### Scenario 1: API Response Handling

You're calling an external API and processing the response.

```python
import json
import httpx

def fetch_user(user_id: int) -> dict:
    """Fetch user from API, handle JSON parsing safely."""
    
    response = httpx.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()  # Raise on 4xx/5xx
    
    # response.text is a string, we need to parse it
    # But httpx (and requests) do this for us:
    return response.json()  # Calls json.loads() internally

def fetch_user_manual(user_id: int) -> dict:
    """Same thing, but showing what .json() does internally."""
    
    response = httpx.get(f"https://api.example.com/users/{user_id}")
    response.raise_for_status()
    
    # This is what .json() does:
    text = response.text  # Get response body as string
    data = json.loads(text)  # Parse JSON into Python dict
    return data
```

**What's actually happening:**

1. HTTP response arrives as bytes
2. Bytes decoded to string (using charset from headers, usually UTF-8)
3. `json.loads()` parses string into Python objects
4. You get a dict (or list, depending on the API)

**Handling parse errors:**

```python
def safe_fetch(url: str) -> dict | None:
    """Fetch JSON with error handling."""
    
    try:
        response = httpx.get(url, timeout=10.0)
        response.raise_for_status()
        return response.json()
    
    except httpx.HTTPStatusError as e:
        # Server returned 4xx or 5xx
        print(f"HTTP error: {e.response.status_code}")
        return None
    
    except json.JSONDecodeError as e:
        # Response wasn't valid JSON
        print(f"Invalid JSON at position {e.pos}: {e.msg}")
        return None
```

### Scenario 2: Configuration File

Loading and saving application configuration.

```python
import json
from pathlib import Path

CONFIG_PATH = Path("config.json")

def load_config() -> dict:
    """Load config from file, with defaults if missing."""
    
    defaults = {
        "debug": False,
        "max_retries": 3,
        "timeout": 30.0,
    }
    
    if not CONFIG_PATH.exists():
        return defaults
    
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        user_config = json.load(f)  # Note: load(), not loads()
    
    # Merge: user config overrides defaults
    return {**defaults, **user_config}


def save_config(config: dict) -> None:
    """Save config to file, human-readable format."""
    
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(
            config,
            f,
            indent=2,           # Pretty-print with 2-space indent
            ensure_ascii=False, # Allow Unicode characters
        )
```

**The difference between `load/dump` and `loads/dumps`:**

| Function | Input/Output | Use Case |
|----------|--------------|----------|
| `json.loads(string)` | String → Python | Parsing API response, string from anywhere |
| `json.dumps(obj)` | Python → String | Building request body, logging |
| `json.load(file)` | File → Python | Reading config/data files |
| `json.dump(obj, file)` | Python → File | Writing config/data files |

The `s` stands for "string." Without `s`, it works with file objects directly.

### Scenario 3: Custom Objects (The Hard Part)

You have a dataclass or custom class that you want to serialize.

```python
import json
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

@dataclass
class User:
    name: str
    email: str
    created_at: datetime
    
# Naive attempt:
user = User("Jay", "jay@example.com", datetime.now())
json.dumps(user)  # TypeError: Object of type User is not JSON serializable
```

**Solution 1: Convert to dict first**

```python
# For dataclasses, use asdict()
user_dict = asdict(user)
# But wait—datetime is still inside!
json.dumps(user_dict)  # Still fails: datetime not serializable
```

**Solution 2: Custom encoder function**

```python
def serialize(obj: Any) -> Any:
    """Convert non-serializable objects to serializable form."""
    
    if isinstance(obj, datetime):
        return obj.isoformat()  # "2024-01-15T10:30:00"
    
    if hasattr(obj, "__dict__"):
        return obj.__dict__  # For regular classes
    
    raise TypeError(f"Cannot serialize {type(obj)}")

# Use as the 'default' parameter:
json.dumps(asdict(user), default=serialize)
# '{"name": "Jay", "email": "jay@example.com", "created_at": "2024-01-15T10:30:00"}'
```

**Solution 3: Custom JSONEncoder class**

For more control, subclass JSONEncoder:

```python
class CustomEncoder(json.JSONEncoder):
    """Encoder that handles datetime and dataclasses."""
    
    def default(self, obj):
        if isinstance(obj, datetime):
            return {"__datetime__": obj.isoformat()}
        
        if hasattr(obj, "__dataclass_fields__"):
            return {"__dataclass__": obj.__class__.__name__, **asdict(obj)}
        
        # Let the base class raise TypeError for unknown types
        return super().default(obj)

# Usage:
json.dumps(user, cls=CustomEncoder)
```

**Solution 4: Custom decoder for round-trip**

To get your objects BACK, you need a custom decoder:

```python
def deserialize(dct: dict) -> Any:
    """Convert special markers back to Python objects."""
    
    if "__datetime__" in dct:
        return datetime.fromisoformat(dct["__datetime__"])
    
    if "__dataclass__" in dct:
        class_name = dct.pop("__dataclass__")
        if class_name == "User":
            return User(**dct)
    
    return dct

# Usage:
text = json.dumps(user, cls=CustomEncoder)
recovered = json.loads(text, object_hook=deserialize)
# recovered is a User object again!
```

### Production Patterns

#### Pattern 1: Safe JSON parsing with validation

```python
import json
from typing import TypeVar, Type
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)

def parse_json_as(text: str, model: Type[T]) -> T:
    """Parse JSON and validate against a Pydantic model."""
    
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e.msg}") from e
    
    try:
        return model.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Validation failed: {e}") from e

# Usage:
from pydantic import BaseModel

class UserResponse(BaseModel):
    id: int
    name: str
    email: str

user = parse_json_as('{"id": 1, "name": "Jay", "email": "j@x.com"}', UserResponse)
```

#### Pattern 2: Streaming large files

Don't load huge JSON files into memory at once:

```python
import json
from typing import Iterator

def iter_json_lines(path: str) -> Iterator[dict]:
    """Read JSON Lines format (one JSON object per line)."""
    
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Skipping invalid JSON on line {line_num}: {e.msg}")

# Usage:
for record in iter_json_lines("huge_data.jsonl"):
    process(record)  # Memory: only one record at a time
```

#### Pattern 3: Deterministic serialization (for hashing/caching)

```python
import json
import hashlib

def stable_json_hash(obj: dict) -> str:
    """Generate consistent hash regardless of key order."""
    
    # sort_keys ensures {"a": 1, "b": 2} and {"b": 2, "a": 1} 
    # produce identical JSON strings
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()
```

### What Breaks: Common Mistakes

**1. Circular references**

```python
a = {"name": "a"}
b = {"name": "b", "friend": a}
a["friend"] = b  # Circular!

json.dumps(a)  # ValueError: Circular reference detected
```

**2. Non-string keys silently converted**

```python
data = {1: "one", 2: "two"}
text = json.dumps(data)  # '{"1": "one", "2": "two"}'
recovered = json.loads(text)  # {"1": "one", "2": "two"} — keys are strings now!
```

**3. Float precision loss**

```python
data = {"value": 0.1 + 0.2}  # 0.30000000000000004
text = json.dumps(data)
recovered = json.loads(text)
print(recovered["value"] == 0.3)  # False!
```

**4. Binary data cannot be serialized**

```python
data = {"image": open("photo.jpg", "rb").read()}
json.dumps(data)  # TypeError: Object of type bytes is not JSON serializable

# Solution: base64 encode
import base64
data = {"image": base64.b64encode(open("photo.jpg", "rb").read()).decode()}
json.dumps(data)  # Works, but the string is huge
```

---

## Summary: The Mental Checklist

When working with JSON, ask:

1. **Am I serializing or deserializing?** → `dumps`/`dump` vs `loads`/`load`

2. **String or file?** → With `s` = string, without = file object

3. **Are all my types JSON-native?** → If not, need `default=` or custom encoder

4. **Do I need round-trip fidelity?** → If yes, need custom encoder AND decoder with markers

5. **Is the data trusted?** → If not, validate after parsing (use Pydantic or similar)

6. **Is the file huge?** → Consider JSON Lines format and streaming
