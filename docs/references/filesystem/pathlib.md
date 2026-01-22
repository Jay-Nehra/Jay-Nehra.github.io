# Python's pathlib Module

## Part 1: Architecture

### The Mental Model: Paths as Objects, Not Strings

The old way treated file paths as strings:
```python
path = "/home/jay/data/file.txt"
directory = path.rsplit("/", 1)[0]  # Ugh
```

The `pathlib` way treats paths as **objects with behavior**:
```python
from pathlib import Path
path = Path("/home/jay/data/file.txt")
directory = path.parent  # Clean
```

**The shift**: From "manipulating text that happens to represent a location" to "asking a Path object questions about itself."

### What Problem Does This Solve?

**Problem 1: String manipulation is error-prone**

```python
# Old way: os.path
import os.path

base = "/home/jay"
filename = "data.txt"
full_path = os.path.join(base, filename)  # Need to remember the function

# What if base ends with "/"? What if filename starts with "/"?
# os.path.join handles it, but you have to know to use it
```

```python
# Even worse: manual string ops
full_path = base + "/" + filename  # Breaks on Windows (\)
```

**Problem 2: Cross-platform nightmares**

```python
# This works on Linux/Mac
path = "/home/jay/data.txt"

# Windows uses backslashes
path = "C:\\Users\\jay\\data.txt"

# Which do you write in your code?
```

**Problem 3: Scattered functionality**

Old way required importing multiple things:
```python
import os
import os.path
import glob
import shutil

os.path.exists(path)
os.path.isfile(path)
os.path.dirname(path)
os.listdir(directory)
glob.glob("*.txt")
os.makedirs(path)
shutil.copy(src, dst)
```

`pathlib` consolidates most of this into one object.

### The Machinery: What Actually Happens

#### Path Object Creation

When you write `Path("/home/jay/data.txt")`:

1. **Platform detection**: Python checks if you're on Windows or POSIX (Unix/Linux/Mac)
2. **Class selection**: Returns `WindowsPath` or `PosixPath` (you don't see this)
3. **Parts parsing**: The string is split into components and stored
4. **No filesystem access yet**: The path object is created even if the file doesn't exist

```python
from pathlib import Path

p = Path("/nonexistent/fake/path.txt")  # Works fine!
print(p.parts)  # ('/', 'nonexistent', 'fake', 'path.txt')
# The filesystem was never touched
```

**Critical insight**: A `Path` object is NOT a file handle. It's a representation of a location. The location might not exist.

#### Pure Paths vs Concrete Paths

```
Path (abstract)
├── PurePath (no filesystem operations)
│   ├── PurePosixPath
│   └── PureWindowsPath
│
└── Path (with filesystem operations)
    ├── PosixPath
    └── WindowsPath
```

**Pure paths**: Just path manipulation, no disk access. Use when:
- Manipulating paths for a DIFFERENT operating system
- Testing without touching disk

**Concrete paths** (what you normally use): Can read/write/check files.

#### The `/` Operator Magic

```python
base = Path("/home/jay")
full = base / "data" / "file.txt"  # Path("/home/jay/data/file.txt")
```

This works because `Path` implements `__truediv__`:
```python
class Path:
    def __truediv__(self, other):
        return Path(os.path.join(str(self), str(other)))
```

It's syntactic sugar for `os.path.join`, but reads better.

#### Filesystem Methods: What They Actually Do

When you call `path.exists()`:

1. Python calls the OS system call `stat()` on the path
2. If the call succeeds, the file/directory exists
3. If it raises "file not found," `exists()` returns `False`

```python
path = Path("/home/jay/file.txt")

path.exists()      # stat() call - does this inode exist?
path.is_file()     # stat() call + check if it's a regular file
path.is_dir()      # stat() call + check if it's a directory
path.stat()        # stat() call, return full metadata (size, times, etc.)
```

**Performance implication**: Each method call is a system call. If you need multiple pieces of info:

```python
# Slow: 3 system calls
if path.exists() and path.is_file() and path.stat().st_size > 0:
    ...

# Fast: 1 system call
try:
    stat = path.stat()
    if stat.st_size > 0:  # Implies exists AND is file (if no error)
        ...
except FileNotFoundError:
    pass
```

### Key Concepts (Behavioral Definitions)

**Path Object**
- What we might assume: "A file handle or file reference"
- What it actually means: A representation of a filesystem location, without any open connection to that location
- Why this matters: We can create paths to nonexistent files, manipulate them, and only touch disk when we explicitly ask

**Immutability**
- What we might assume: "We can modify a path object"
- What it actually means: Path objects are immutable. Methods like `with_suffix()` return NEW paths.
- Why this matters: Safe to share paths across threads, store in sets/dicts

**Platform Abstraction**
- What we might assume: "We need to handle Windows paths differently"
- What it actually means: `Path()` automatically uses the right type; `/` works on all platforms
- Why this matters: Write code once, run anywhere (for path logic)

**Lazy Evaluation**
- What we might assume: "`Path(x)` validates that x exists"
- What it actually means: Path objects are created without touching the filesystem
- Why this matters: Fast to create, but we must explicitly check existence if we care

### Design Decisions: Why Is It This Way?

**Why objects instead of functions?**

Chaining is cleaner:
```python
# Function style (old)
os.path.splitext(os.path.basename(path))[0]

# Object style (pathlib)
path.stem
```

And you get autocomplete in your editor.

**Why immutable?**

Paths represent locations, not files. Locations don't change. If you want a different location, you get a different object:
```python
original = Path("/data/file.txt")
renamed = original.with_name("other.txt")  # New object
# original is unchanged
```

**Why `/` for joining?**

It reads like a filesystem path:
```python
result = base / subdir / filename  # Looks like a path
result = os.path.join(base, subdir, filename)  # Looks like a function call
```

The `/` operator was available (not commonly used for division with strings) and visually matches path separators.

### What Breaks If You Misunderstand

**Mistake 1: Assuming existence**

```python
path = Path("/data/file.txt")
content = path.read_text()  # FileNotFoundError if doesn't exist!

# Fix: Check first, or handle exception
if path.exists():
    content = path.read_text()
```

**Mistake 2: Forgetting immutability**

```python
path = Path("/data/file.txt")
path.with_suffix(".json")  # Returns new Path, doesn't modify!
print(path)  # Still /data/file.txt

# Fix: Assign the result
path = path.with_suffix(".json")
```

**Mistake 3: Mixing strings and paths incorrectly**

```python
# Sometimes you need a string (for libraries that don't accept Path)
path = Path("/data/file.txt")
old_library.process(path)  # May fail if library expects str

# Fix: Convert explicitly
old_library.process(str(path))
# Or in Python 3.6+, many libraries accept Path via os.fspath()
```

**Mistake 4: Using wrong slashes in strings**

```python
# This only works on Windows
path = Path("C:\\Users\\jay")  

# This works everywhere (forward slashes work on Windows too)
path = Path("C:/Users/jay")

# This is best (let Path handle it)
path = Path("C:") / "Users" / "jay"
```

---

## Part 2: Scenarios

### Scenario 1: Basic File Operations

Reading, writing, and checking files:

```python
from pathlib import Path

# Creating paths
data_dir = Path("/home/jay/data")
file_path = data_dir / "input.txt"

# Checking existence
if not file_path.exists():
    print(f"File not found: {file_path}")
    
if file_path.is_file():
    print("It's a file")
elif file_path.is_dir():
    print("It's a directory")

# Reading files (returns string)
content = file_path.read_text(encoding="utf-8")

# Reading binary files (returns bytes)
data = file_path.read_bytes()

# Writing files
output_path = data_dir / "output.txt"
output_path.write_text("Hello, World!", encoding="utf-8")

# Appending (need to use open())
with output_path.open("a", encoding="utf-8") as f:
    f.write("\nMore content")
```

**What `read_text()` does internally:**
```python
def read_text(self, encoding=None):
    with self.open("r", encoding=encoding) as f:
        return f.read()
```

It's convenience, not magic.

### Scenario 2: Path Manipulation

Getting parts of paths without touching the filesystem:

```python
from pathlib import Path

path = Path("/home/jay/projects/myapp/data/users.json")

# Components
path.parts      # ('/', 'home', 'jay', 'projects', 'myapp', 'data', 'users.json')
path.parent     # Path('/home/jay/projects/myapp/data')
path.parents    # Sequence of all parents up to root
path.name       # 'users.json' (filename with extension)
path.stem       # 'users' (filename without extension)
path.suffix     # '.json' (extension including dot)
path.suffixes   # ['.json'] (all extensions, e.g., ['.tar', '.gz'])

# Transformations (return NEW paths)
path.with_name("orders.json")      # /home/jay/.../data/orders.json
path.with_stem("customers")        # /home/jay/.../data/customers.json (Python 3.9+)
path.with_suffix(".csv")           # /home/jay/.../data/users.csv

# Absolute vs relative
rel_path = Path("data/file.txt")
abs_path = rel_path.resolve()      # Full absolute path, resolves symlinks
rel_path.absolute()                # Absolute but doesn't resolve symlinks

# Relative to another path
full = Path("/home/jay/project/data/file.txt")
base = Path("/home/jay/project")
full.relative_to(base)             # Path('data/file.txt')
```

### Scenario 3: Directory Operations

Listing, creating, and navigating directories:

```python
from pathlib import Path

project_dir = Path("/home/jay/project")

# Listing contents (non-recursive)
for item in project_dir.iterdir():
    if item.is_file():
        print(f"File: {item.name}")
    elif item.is_dir():
        print(f"Dir:  {item.name}")

# Glob patterns (recursive with **)
for py_file in project_dir.glob("*.py"):        # Current dir only
    print(py_file)

for py_file in project_dir.glob("**/*.py"):     # Recursive
    print(py_file)

for test_file in project_dir.glob("**/test_*.py"):  # All test files
    print(test_file)

# rglob = recursive glob (shorthand)
for py_file in project_dir.rglob("*.py"):       # Same as **/*.py
    print(py_file)

# Creating directories
new_dir = project_dir / "output" / "reports"
new_dir.mkdir()                    # Fails if parent doesn't exist
new_dir.mkdir(parents=True)        # Creates parents too
new_dir.mkdir(parents=True, exist_ok=True)  # No error if exists

# Current working directory
cwd = Path.cwd()

# Home directory
home = Path.home()  # /home/jay or C:\Users\jay
```

### Scenario 4: File Metadata and Comparison

Getting file info and comparing paths:

```python
from pathlib import Path
import time

path = Path("/home/jay/data/file.txt")

# Full stat info
stat = path.stat()
stat.st_size      # Size in bytes
stat.st_mtime     # Modification time (Unix timestamp)
stat.st_ctime     # Creation time (platform-dependent meaning)
stat.st_mode      # File mode/permissions

# Formatted times
mod_time = time.ctime(stat.st_mtime)  # 'Mon Jan 15 10:30:00 2024'

# Comparing paths
path1 = Path("/home/jay/../jay/file.txt")
path2 = Path("/home/jay/file.txt")

path1 == path2           # False! String comparison
path1.resolve() == path2.resolve()  # True! After resolution

# Checking if one path is inside another
child = Path("/home/jay/project/file.txt")
parent = Path("/home/jay/project")
child.is_relative_to(parent)  # True (Python 3.9+)
```

### Scenario 5: Common Project Patterns

Real-world patterns for projects:

```python
from pathlib import Path

# Pattern 1: Finding project root (look for marker file)
def find_project_root(marker: str = "pyproject.toml") -> Path:
    """Walk up from current dir until we find the project marker."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / marker).exists():
            return parent
    raise FileNotFoundError(f"Could not find {marker} in parent directories")

PROJECT_ROOT = find_project_root()
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"

# Pattern 2: Relative to script location
# __file__ is the current script's path
SCRIPT_DIR = Path(__file__).parent.resolve()
CONFIG_PATH = SCRIPT_DIR / "config.json"

# Pattern 3: Temporary files with cleanup
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    tmp_path = Path(tmp)
    work_file = tmp_path / "working.txt"
    work_file.write_text("temporary data")
    # Do work...
# Directory and contents deleted automatically

# Pattern 4: Safe file writing (atomic)
def safe_write(path: Path, content: str) -> None:
    """Write to temp file, then rename (atomic on most filesystems)."""
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(content, encoding="utf-8")
    tmp_path.rename(path)  # Atomic replace

# Pattern 5: Backup before overwrite
def write_with_backup(path: Path, content: str) -> None:
    """Create .bak backup before overwriting."""
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        path.rename(backup)
    path.write_text(content, encoding="utf-8")
```

### Production Patterns

#### Pattern 1: Configuration-driven paths

```python
from pathlib import Path
import os

class Paths:
    """Centralized path configuration."""
    
    # Base from environment or default
    BASE = Path(os.environ.get("APP_BASE_DIR", Path.cwd()))
    
    # Derived paths
    DATA = BASE / "data"
    LOGS = BASE / "logs"
    CACHE = BASE / ".cache"
    
    @classmethod
    def ensure_dirs(cls):
        """Create all required directories."""
        for attr in ["DATA", "LOGS", "CACHE"]:
            getattr(cls, attr).mkdir(parents=True, exist_ok=True)

# At startup
Paths.ensure_dirs()
```

#### Pattern 2: Type-safe path handling

```python
from pathlib import Path
from typing import Iterator

def get_data_files(directory: Path, pattern: str = "*.csv") -> Iterator[Path]:
    """Yield data files matching pattern."""
    if not directory.is_dir():
        raise ValueError(f"Not a directory: {directory}")
    
    yield from directory.glob(pattern)

def process_file(file_path: Path) -> dict:
    """Process a single file."""
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    content = file_path.read_text(encoding="utf-8")
    # Process...
    return {"path": str(file_path), "size": file_path.stat().st_size}
```

#### Pattern 3: Cross-platform path handling

```python
from pathlib import Path, PurePosixPath, PureWindowsPath

def normalize_path_from_config(raw_path: str) -> Path:
    """Handle paths from config that might be from different OS."""
    
    # Detect Windows-style path
    if "\\" in raw_path or (len(raw_path) > 1 and raw_path[1] == ":"):
        # Parse as Windows, convert to current platform
        parts = PureWindowsPath(raw_path).parts
    else:
        parts = PurePosixPath(raw_path).parts
    
    # Reconstruct for current platform
    return Path(*parts)
```

### What Breaks: Common Mistakes

**1. Not handling encoding**

```python
# May fail on non-ASCII content with default encoding
content = path.read_text()  # Uses locale encoding

# Fix: Always specify encoding
content = path.read_text(encoding="utf-8")
```

**2. Modifying paths during iteration**

```python
# DANGEROUS: Modifying while iterating
for file in directory.iterdir():
    file.unlink()  # May cause issues

# Fix: Collect first, then modify
files = list(directory.iterdir())
for file in files:
    file.unlink()
```

**3. Assuming relative paths are relative to script**

```python
# In script at /home/jay/project/src/script.py
data = Path("data/file.txt").read_text()  
# Looks for ./data/file.txt relative to CWD, not script!

# Fix: Use __file__
script_dir = Path(__file__).parent
data = (script_dir / "data/file.txt").read_text()
```

**4. Race conditions**

```python
# DANGEROUS: Time-of-check to time-of-use race
if path.exists():
    content = path.read_text()  # File might be deleted between check and read!

# Fix: Just try and handle the exception
try:
    content = path.read_text()
except FileNotFoundError:
    content = None
```

---

## Summary: The Mental Checklist

1. **Am I manipulating the path or accessing the file?**
   - Manipulation (parent, stem, suffix): No disk access
   - Access (read, write, exists): Hits the filesystem

2. **Do I need cross-platform compatibility?**
   - Use `/` operator, not string concatenation
   - Never hardcode `\` or `/`

3. **Does the path need to exist?**
   - Creating a Path object doesn't check existence
   - Use `exists()`, `is_file()`, `is_dir()` to check

4. **Am I handling errors?**
   - `FileNotFoundError` for missing files
   - `PermissionError` for access issues
   - `IsADirectoryError` / `NotADirectoryError` for type mismatches

5. **String or Path?**
   - Modern libraries accept Path directly
   - Old libraries need `str(path)`
   - When in doubt, `str()` works everywhere
