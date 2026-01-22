---
tags:
  - python
  - config
  - environment
---

# Environment Variables & python-dotenv

## Part 1: Architecture

### The Mental Model: The Operating System's Sticky Notes

Before our Python program even starts, the operating system has already prepared a sheet of sticky notes called the **environment**. These notes contain key-value pairs that any program can read.

When we run a Python script, the OS hands your process a *copy* of these sticky notes. Our program can:
- Read any note
- Add new notes (for itself and any child processes it spawns)
- Modify notes (but only your copy—other processes don't see it)

`os.environ` is Python's window into this sticky-note sheet.

### What Problem Does This Solve?

**The fundamental problem**: Our code needs to behave differently in different contexts.

- Local development: connect to localhost database
- Staging: connect to staging database
- Production: connect to production database with real credentials

**The naive approaches and why they fail:**

**Approach 1: Hardcode values**
```python
DATABASE_URL = "postgresql://localhost:5432/mydb"  # What about production?
```
Problem: You have to change code to deploy. Code changes require testing. This is slow and dangerous.

**Approach 2: Config file checked into git**
```python
# config.py
DATABASE_URL = "postgresql://prod-server:5432/mydb"
SECRET_KEY = "super-secret-123"  # EXPOSED IN GIT HISTORY FOREVER
```
Problem: Secrets in version control. Once pushed, they're compromised even if deleted later.

**Approach 3: Config file NOT in git**
```python
# config.py (in .gitignore)
DATABASE_URL = "..."
```
Problem: How does this file get to production? Manual copying? Now you have deployment complexity.

**The environment variable solution:**

1. Your code reads from `os.environ` (no secrets in code)
2. In development, you set variables via a `.env` file (convenient, gitignored)
3. In production, the platform sets variables (AWS, Heroku, Kubernetes all support this)

**The key insight**: Configuration lives OUTSIDE your code, injected at runtime by the environment.

### The Machinery: What Actually Happens

#### The OS-Level Foundation

When you open a terminal and run `python script.py`, here's the chain:

1. **Shell has environment**: Your terminal (bash, zsh, PowerShell) has its own environment variables
2. **Fork**: The shell creates a child process for Python
3. **Copy**: The child process gets a COPY of the parent's environment
4. **Execution**: Python starts, reads the copied environment into `os.environ`

```
Terminal (bash)
├── HOME=/Users/jay
├── PATH=/usr/bin:/usr/local/bin
├── DATABASE_URL=postgres://localhost/dev
│
└── [spawns] python script.py
    └── os.environ gets a COPY:
        ├── HOME=/Users/jay
        ├── PATH=/usr/bin:/usr/local/bin
        ├── DATABASE_URL=postgres://localhost/dev
```

**Critical implications:**

1. **Changes don't propagate up**: If Python modifies `os.environ`, the parent shell doesn't see it
2. **Changes DO propagate down**: If Python spawns a subprocess, that subprocess inherits Python's modified environment
3. **It's a copy, not a reference**: Reading is cheap, the OS already copied everything at process start

#### What is `os.environ` exactly?

It's a dict-like object, but NOT a regular dict:

```python
import os

# Reading (like a dict)
value = os.environ["HOME"]  # Raises KeyError if missing
value = os.environ.get("HOME")  # Returns None if missing
value = os.environ.get("HOME", "/default")  # With default

# Writing (THIS IS WHERE IT'S DIFFERENT)
os.environ["MY_VAR"] = "my_value"
```

When you write to `os.environ`, Python doesn't just update an internal dict. It calls the C library function `putenv()`, which updates the actual process environment in the OS.

**Why does this matter?**

```python
os.environ["API_KEY"] = "secret123"

import boto3  # AWS library
# boto3 checks os.environ["AWS_ACCESS_KEY_ID"] internally
# It sees your changes because you modified the REAL environment
```

If `os.environ` were a regular dict, other libraries wouldn't see your changes.

#### Where dotenv Fits In

`python-dotenv` solves the "development convenience" problem.

**The problem**: In production, environment variables are set by the platform. But locally, you'd have to:
```bash
export DATABASE_URL=postgres://localhost/dev
export SECRET_KEY=dev-secret
export DEBUG=true
python app.py
```

Every time. In every terminal. That's annoying.

**The solution**: A `.env` file:
```
# .env (gitignored!)
DATABASE_URL=postgres://localhost/dev
SECRET_KEY=dev-secret
DEBUG=true
```

And in your code:
```python
from dotenv import load_dotenv
load_dotenv()  # Reads .env file, calls os.environ[key] = value for each line
```

**What `load_dotenv()` actually does:**

1. Find `.env` file (current directory, or walk up to project root)
2. Parse each line: `KEY=value`
3. For each parsed pair, call `os.environ.setdefault(key, value)`

**The critical behavior: "First One Wins"**

```python
# If DATABASE_URL is already set in the shell...
load_dotenv()  # ...the .env file value is IGNORED (setdefault)
```

This is **intentional and important**:
- In production, the platform sets real values
- `load_dotenv()` runs but does nothing (values already exist)
- Your code reads from `os.environ` and gets production values

The `.env` file only "fills in" what's missing.

### Key Concepts (Behavioral Definitions)

**Environment Variable**
- What we might assume: "A Python variable stored somewhere"
- What it actually means: A key-value pair in the OS process's memory, inherited from parent process, accessible via system calls
- Why this matters: It's not Python-specific; any library in any language in our process can read these

**Process Isolation**
- What we might assume: "If we set a variable, everyone sees it"
- What it actually means: Each process has its own copy; changes are invisible to parent or sibling processes
- Why this matters: We can't "break" other running programs by modifying our environment

**Injection**
- What we might assume: "`load_dotenv()` creates variables"
- What it actually means: It reads a file and calls `os.environ[k] = v` for each entry—the same as if we typed `export K=v` in the shell
- Why this matters: There's nothing magical; we could do this with a for loop

**Override vs Default**
- What we might assume: "`load_dotenv()` always uses our .env file"
- What it actually means: By default, existing variables are NOT overwritten (`setdefault` behavior)
- Why this matters: Production config survives even if a .env file accidentally exists

### Design Decisions: Why Is It This Way?

**Why environment variables instead of a config file format?**

1. **Universal**: Every OS, every language, every platform supports them
2. **No parsing code**: Just read a string, no YAML/JSON parsing needed
3. **12-Factor App standard**: Industry consensus for cloud deployment
4. **Platform integration**: AWS, Heroku, Docker, Kubernetes all inject config this way

**Why is `load_dotenv()` not automatic?**

Explicit is better than implicit. If it ran on import, you'd have:
- Hidden file reads on every import
- No control over WHEN it runs
- Harder debugging ("where did this value come from?")

**Why does dotenv default to NOT overriding?**

The "infrastructure wins" principle:
- Production config is set by ops/platform (AWS Secrets Manager, Kubernetes)
- Code-level config (`.env` file) is developer convenience
- If there's a conflict, trust infrastructure, not the file

If you need override behavior:
```python
load_dotenv(override=True)  # File wins over existing
```

### What Breaks If You Misunderstand

**Mistake 1: Expecting variables to persist**

```python
# script_a.py
import os
os.environ["MY_VAR"] = "hello"

# script_b.py (run separately)
import os
print(os.environ.get("MY_VAR"))  # None! Different process!
```

**Mistake 2: Committing secrets**

```python
# .env
API_KEY=sk-real-production-key

# You forgot to add .env to .gitignore
# Now your key is in git history forever
```

**Mistake 3: Deploying with override=True**

```python
# You're debugging locally and add:
load_dotenv(override=True)

# You deploy. The old .env file somehow exists on the server.
# Production secrets are overwritten by stale dev values.
# Your app connects to the dev database in production.
```

**Mistake 4: Reading before loading**

```python
import os
DATABASE_URL = os.environ["DATABASE_URL"]  # KeyError in development!

from dotenv import load_dotenv
load_dotenv()  # Too late
```

---

## Part 2: Scenarios

### Scenario 1: Basic Application Setup

The correct pattern for any Python application:

```python
# app.py
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env FIRST, before reading any config
# Only affects development—in production, variables are already set
load_dotenv()

# Now read config
class Config:
    DATABASE_URL: str = os.environ["DATABASE_URL"]  # Required
    SECRET_KEY: str = os.environ["SECRET_KEY"]  # Required
    DEBUG: bool = os.environ.get("DEBUG", "false").lower() == "true"  # Optional
    LOG_LEVEL: str = os.environ.get("LOG_LEVEL", "INFO")  # Optional with default

# Use it
config = Config()
```

**Your `.env` file (gitignored):**
```
DATABASE_URL=postgresql://localhost:5432/myapp_dev
SECRET_KEY=dev-secret-not-for-production
DEBUG=true
```

**Your `.env.example` file (committed to git, no secrets):**
```
DATABASE_URL=postgresql://localhost:5432/myapp_dev
SECRET_KEY=change-me-in-production
DEBUG=false
```

### Scenario 2: Multiple Environments

You want different configs for dev, test, and staging:

```python
# config.py
import os
from pathlib import Path
from dotenv import load_dotenv

def load_config(env_name: str | None = None):
    """Load configuration for the specified environment."""
    
    # Determine which env
    env = env_name or os.environ.get("APP_ENV", "development")
    
    # Try environment-specific file first
    env_file = Path(f".env.{env}")
    if env_file.exists():
        load_dotenv(env_file)
    
    # Fall back to default .env
    load_dotenv()  # Won't override what's already set

# Usage:
# Development: just run, reads .env
# Staging: APP_ENV=staging python app.py (reads .env.staging first)
```

**File structure:**
```
project/
├── .env              # Local development (gitignored)
├── .env.example      # Template (committed)
├── .env.test         # Test settings (maybe committed, no secrets)
└── .env.staging      # Staging settings (gitignored)
```

### Scenario 3: Required vs Optional Variables

Handle missing required variables gracefully:

```python
import os
import sys
from dotenv import load_dotenv

load_dotenv()

def require_env(name: str) -> str:
    """Get required environment variable or exit with helpful message."""
    value = os.environ.get(name)
    if value is None:
        print(f"ERROR: Required environment variable {name} is not set")
        print(f"  Add it to your .env file or export it in your shell")
        print(f"  Example: export {name}=your-value")
        sys.exit(1)
    return value

def get_env(name: str, default: str = "") -> str:
    """Get optional environment variable with default."""
    return os.environ.get(name, default)

def get_env_bool(name: str, default: bool = False) -> bool:
    """Get boolean environment variable."""
    value = os.environ.get(name, str(default)).lower()
    return value in ("true", "1", "yes", "on")

def get_env_int(name: str, default: int = 0) -> int:
    """Get integer environment variable."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"WARNING: {name}={value} is not a valid integer, using default {default}")
        return default

# Usage:
DATABASE_URL = require_env("DATABASE_URL")  # Fails fast if missing
DEBUG = get_env_bool("DEBUG", default=False)
WORKER_COUNT = get_env_int("WORKER_COUNT", default=4)
```

### Scenario 4: Pydantic Settings (Production Pattern)

For larger applications, use Pydantic for validation:

```python
# settings.py
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    """Application settings with validation."""
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,  # DATABASE_URL == database_url
    )
    
    # Required
    database_url: str
    secret_key: str
    
    # Optional with defaults
    debug: bool = False
    log_level: str = "INFO"
    worker_count: int = 4
    
    # Computed/derived
    @property
    def is_production(self) -> bool:
        return not self.debug

# Usage
settings = Settings()  # Reads from env + .env, validates types
print(settings.database_url)  # Type-safe, validated
```

**What Pydantic Settings does:**

1. Reads from `os.environ` (priority)
2. Falls back to `.env` file
3. Validates types (is `worker_count` really an int?)
4. Provides defaults
5. Fails fast with clear errors if invalid

### Scenario 5: Subprocess Environment

Passing environment to child processes:

```python
import os
import subprocess

# Child process inherits your environment automatically
subprocess.run(["python", "worker.py"])  
# worker.py sees everything in os.environ

# Pass ADDITIONAL variables
subprocess.run(
    ["python", "worker.py"],
    env={**os.environ, "WORKER_ID": "1"}  # Inherit + add
)

# Pass ONLY specific variables (dangerous—child may need PATH, etc.)
subprocess.run(
    ["python", "worker.py"],
    env={"DATABASE_URL": os.environ["DATABASE_URL"]}  # Only this!
)
```

### Production Patterns

#### Pattern 1: Early validation

```python
# main.py
import sys
from config import Settings

def main():
    try:
        settings = Settings()
    except Exception as e:
        print(f"Configuration error: {e}")
        print("Check your environment variables and .env file")
        sys.exit(1)
    
    # Settings valid, start application
    app = create_app(settings)
    app.run()
```

#### Pattern 2: No secrets in logs

```python
import os
import logging

# NEVER do this
logging.info(f"Connecting with {os.environ['DATABASE_URL']}")  # Leaks password!

# Do this instead
logging.info("Connecting to database")  # Or mask it
```

#### Pattern 3: Docker/container setup

```dockerfile
# Dockerfile - don't bake in secrets
FROM python:3.12
COPY . /app
WORKDIR /app
# ENV DATABASE_URL=... # DON'T DO THIS

# Run with secrets injected:
# docker run -e DATABASE_URL=... -e SECRET_KEY=... myapp
```

### What Breaks: Common Mistakes

**1. Reading before load**

```python
import os
DATABASE_URL = os.environ["DATABASE_URL"]  # Module-level, runs at import
from dotenv import load_dotenv
load_dotenv()  # Too late
```

Fix: Load first, or use lazy access.

**2. Type confusion**

```python
DEBUG = os.environ.get("DEBUG", False)  # WRONG: default is bool, but...
# If DEBUG=false in .env, you get the STRING "false", which is truthy!

# Fix:
DEBUG = os.environ.get("DEBUG", "").lower() in ("true", "1", "yes")
```

**3. Stale variables after code change**

```python
# You rename API_KEY to SERVICE_TOKEN in code
# But your .env still has API_KEY=...
# And production still has API_KEY set
# Result: KeyError in production
```

**4. Case sensitivity (platform-dependent)**

```python
# Linux: case-sensitive
os.environ["API_Key"] != os.environ["API_KEY"]

# Windows: case-insensitive (usually)
os.environ["API_Key"] == os.environ["API_KEY"]  # Sometimes!
```

Convention: ALWAYS use UPPER_SNAKE_CASE for env vars.

---

## Summary: The Mental Checklist

1. **Where does this variable come from?**
   - Shell? → `export VAR=value` or set in platform
   - File? → `.env` via `load_dotenv()`
   - Parent process? → Inherited automatically

2. **Is this required or optional?**
   - Required: Fail fast if missing
   - Optional: Provide sensible default

3. **What type is it?**
   - Everything is a string—parse explicitly
   - Use Pydantic for validation

4. **Is this a secret?**
   - Yes: Never log, never commit, use secrets manager in production
   - No: Can go in `.env.example`

5. **Load order matters:**
   - `load_dotenv()` BEFORE reading config
   - Module-level reads happen at import time
