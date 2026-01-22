# Python's argparse Module

## Part 1: Architecture

### The Mental Model: The Command Interpreter

When we run a command like:
```bash
python script.py --verbose -n 5 input.txt output.txt
```

Our script receives this as a list of strings:
```python
sys.argv = ['script.py', '--verbose', '-n', '5', 'input.txt', 'output.txt']
```

`argparse` is an **interpreter** that:
1. Takes this list of strings
2. Applies rules you define (what arguments exist, what types, required or optional)
3. Returns a structured object with parsed values

Without it, you're doing string parsing by hand. With it, you declare what you expect and let the library handle the messy details.

### What Problem Does This Solve?

**The naive approach:**

```python
import sys

# Manual parsing — fragile and tedious
if len(sys.argv) < 3:
    print("Usage: script.py input output")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

verbose = '--verbose' in sys.argv or '-v' in sys.argv

# What about -n? Where is it? What's after it?
# What if user writes -n5 vs -n 5?
# What if they put --verbose between -n and 5?
```

This gets ugly fast. Every edge case requires more code.

**What argparse gives you:**

1. **Declaration over parsing**: Say what you want, not how to find it
2. **Automatic help**: `--help` generated for free
3. **Type conversion**: String "5" → integer 5
4. **Validation**: Required arguments, choices, ranges
5. **Flexible syntax**: `-n 5`, `-n5`, `--number=5` all work

### The Machinery: What Actually Happens

#### The Parsing Process

```python
import argparse

parser = argparse.ArgumentParser(description="Process files")
parser.add_argument("input", help="Input file")
parser.add_argument("output", help="Output file")
parser.add_argument("-n", "--number", type=int, default=1)
parser.add_argument("-v", "--verbose", action="store_true")

args = parser.parse_args()  # ← All the magic happens here
```

When `parse_args()` runs:

**Step 1: Tokenization**
```
sys.argv = ['script.py', '--verbose', '-n', '5', 'input.txt', 'output.txt']
              ↓
Tokens:     ['--verbose', '-n', '5', 'input.txt', 'output.txt']
            (script name removed)
```

**Step 2: Classification**

The parser has two types of arguments:
- **Positional**: No dash prefix, order matters (`input`, `output`)
- **Optional**: Dash prefix, can appear anywhere (`-n`, `--verbose`)

```
'--verbose'  → Optional, matches --verbose
'-n'         → Optional, matches -n/--number
'5'          → Value for -n (because -n expects a value)
'input.txt'  → Positional #1 (input)
'output.txt' → Positional #2 (output)
```

**Step 3: Value Processing**

For each argument:
1. Find matching rule in parser
2. Apply `type=` converter (string → target type)
3. Apply `action=` (store, store_true, append, etc.)
4. Check constraints (choices, required)

**Step 4: Namespace Construction**

```python
args = Namespace(
    input='input.txt',
    output='output.txt',
    number=5,
    verbose=True
)

# Access as attributes
args.input   # 'input.txt'
args.number  # 5 (int, not string!)
```

#### Positional vs Optional: The Core Distinction

```python
# Positional: No dashes, filled in order
parser.add_argument("filename")     # Required, position 1
parser.add_argument("destination")  # Required, position 2

# Optional: Has dashes, can appear anywhere
parser.add_argument("-v", "--verbose")  # Flag
parser.add_argument("-n", "--number")   # Takes a value
```

**Key insight**: Positional arguments are matched by position, not name. The name is just for the attribute.

```python
# User types: script.py foo bar
# Parser sees: positional_1='foo', positional_2='bar'
# You access: args.filename='foo', args.destination='bar'
```

#### Actions: What To Do With the Value

The `action=` parameter controls how arguments are processed:

| Action | What it does | Use case |
|--------|--------------|----------|
| `store` (default) | Store the value | Most arguments |
| `store_true` | Store `True` if present | Boolean flags (`--verbose`) |
| `store_false` | Store `False` if present | Inverse flags (`--no-cache`) |
| `store_const` | Store a constant value | Preset options |
| `append` | Append to a list | Repeatable options (`-i file1 -i file2`) |
| `count` | Count occurrences | Verbosity levels (`-vvv`) |

```python
# store_true: Flag that sets True
parser.add_argument("--verbose", action="store_true")
# --verbose → args.verbose = True
# (nothing) → args.verbose = False

# append: Collect multiple values
parser.add_argument("-i", "--include", action="append")
# -i foo -i bar → args.include = ['foo', 'bar']

# count: Count repetitions
parser.add_argument("-v", action="count", default=0)
# -v    → args.v = 1
# -vvv  → args.v = 3
```

### Key Concepts (Behavioral Definitions)

**Positional Argument**
- What we might assume: "An argument that must be in a specific position"
- What it actually means: An argument without dashes, consumed in declaration order
- Why this matters: Position of positionals relative to each other matters, but optionals can appear anywhere between them

**Optional Argument**
- What we might assume: "An argument we can skip"
- What it actually means: An argument with dash prefix (`-x` or `--xxx`), identified by name not position
- Why this matters: "Optional" means "identified by flag," not necessarily "not required"

**Namespace**
- What we might assume: "A dictionary"
- What it actually means: A simple object where arguments become attributes
- Why this matters: Access via `args.name`, not `args["name"]`

**Action**
- What we might assume: "What to do after parsing"
- What it actually means: How to transform/store the parsed value
- Why this matters: Different actions for flags vs values vs collections

### Design Decisions: Why Is It This Way?

**Why attributes instead of a dictionary?**

```python
# Dictionary style (rejected)
args["verbose"]

# Attribute style (chosen)
args.verbose
```

Attributes are:
- Easier to type (no quotes, no brackets)
- Autocomplete-friendly in editors
- Feel more like accessing properties

**Why separate positional and optional?**

They solve different problems:
- Positional: Few, essential, always provided (file to process)
- Optional: Many, modifiers, often have defaults (flags, settings)

Mixing them would force awkward syntax or complex rules.

**Why `store_true` instead of `type=bool`?**

```python
# This doesn't work as expected:
parser.add_argument("--verbose", type=bool)
# --verbose false → args.verbose = True!
# Because bool("false") == True (non-empty string)

# This is why store_true exists:
parser.add_argument("--verbose", action="store_true")
# --verbose → args.verbose = True
# (absent) → args.verbose = False
```

### What Breaks If You Misunderstand

**Mistake 1: Using type=bool**

```python
parser.add_argument("--flag", type=bool)
# --flag true  → True (ok)
# --flag false → True (!!! bool("false") is True)

# Fix: Use store_true/store_false or a custom type
parser.add_argument("--flag", action="store_true")
```

**Mistake 2: Positional after optional with nargs**

```python
parser.add_argument("-f", "--files", nargs="+")  # One or more
parser.add_argument("output")  # Positional

# python script.py -f a b c d
# Is 'd' a file or the output? Ambiguous!

# Fix: Put positionals first, or use --
# python script.py output -f a b c
# python script.py -f a b c -- output
```

**Mistake 3: Forgetting nargs for optional lists**

```python
parser.add_argument("-f", "--files")  # Only takes ONE value!
# -f a b c → 'a', and b,c are errors

# Fix: Use nargs
parser.add_argument("-f", "--files", nargs="+")  # One or more
parser.add_argument("-f", "--files", nargs="*")  # Zero or more
```

**Mistake 4: Required positionals with defaults**

```python
parser.add_argument("input")  # Required, no default possible
# Can't make positional optional!

# Fix: Make it an optional argument
parser.add_argument("-i", "--input", default="stdin")
# Or use nargs='?'
parser.add_argument("input", nargs='?', default="stdin")
```

---

## Part 2: Scenarios

### Scenario 1: Basic Script with File Arguments

A script that processes an input file:

```python
#!/usr/bin/env python3
"""Process a data file and write results."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(
        description="Process data files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data.csv                 # Process with defaults
  %(prog)s data.csv -o result.json  # Specify output
  %(prog)s data.csv --format yaml   # Different format
        """
    )
    
    # Positional: the input file
    parser.add_argument(
        "input",
        type=Path,  # Automatic conversion to Path object
        help="Input file to process"
    )
    
    # Optional: output file
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Output file (default: stdout)"
    )
    
    # Optional: format choice
    parser.add_argument(
        "-f", "--format",
        choices=["json", "yaml", "csv"],
        default="json",
        help="Output format (default: %(default)s)"
    )
    
    # Flag: verbose mode
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print detailed progress"
    )
    
    args = parser.parse_args()
    
    # Validate input exists
    if not args.input.exists():
        parser.error(f"Input file not found: {args.input}")
    
    # Use the parsed arguments
    if args.verbose:
        print(f"Processing {args.input}")
        print(f"Format: {args.format}")
    
    # ... actual processing ...
    
    if args.output:
        args.output.write_text(result)
    else:
        print(result)

if __name__ == "__main__":
    main()
```

**What this gives you:**

```bash
$ python script.py --help
usage: script.py [-h] [-o OUTPUT] [-f {json,yaml,csv}] [-v] input

Process data files

positional arguments:
  input                 Input file to process

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output file (default: stdout)
  -f {json,yaml,csv}, --format {json,yaml,csv}
                        Output format (default: json)
  -v, --verbose         Print detailed progress

Examples:
  script.py data.csv                 # Process with defaults
  script.py data.csv -o result.json  # Specify output
  script.py data.csv --format yaml   # Different format
```

### Scenario 2: Subcommands (Like git)

For tools with multiple operations:

```python
#!/usr/bin/env python3
"""Database management tool."""

import argparse

def cmd_init(args):
    print(f"Initializing database: {args.name}")
    if args.force:
        print("Forcing recreation")

def cmd_migrate(args):
    print(f"Running migrations up to: {args.target or 'latest'}")

def cmd_status(args):
    print("Checking database status...")

def main():
    parser = argparse.ArgumentParser(
        description="Database management tool"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        help="Command to run"
    )
    
    # init subcommand
    init_parser = subparsers.add_parser("init", help="Initialize database")
    init_parser.add_argument("name", help="Database name")
    init_parser.add_argument("--force", action="store_true")
    init_parser.set_defaults(func=cmd_init)
    
    # migrate subcommand
    migrate_parser = subparsers.add_parser("migrate", help="Run migrations")
    migrate_parser.add_argument(
        "-t", "--target",
        help="Target migration (default: latest)"
    )
    migrate_parser.set_defaults(func=cmd_migrate)
    
    # status subcommand
    status_parser = subparsers.add_parser("status", help="Show status")
    status_parser.set_defaults(func=cmd_status)
    
    args = parser.parse_args()
    args.func(args)  # Call the handler

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
$ python db.py --help
usage: db.py [-h] {init,migrate,status} ...

$ python db.py init mydb --force
Initializing database: mydb
Forcing recreation

$ python db.py migrate --target 003
Running migrations up to: 003
```

### Scenario 3: Multiple Input Files

Processing multiple files with the same script:

```python
#!/usr/bin/env python3
"""Concatenate files."""

import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    
    # Multiple positional arguments
    parser.add_argument(
        "files",
        nargs="+",  # One or more
        type=Path,
        help="Files to concatenate"
    )
    
    # Or use -i for repeated optional
    # parser.add_argument(
    #     "-i", "--input",
    #     action="append",
    #     type=Path,
    #     help="Input file (can repeat)"
    # )
    
    parser.add_argument(
        "-o", "--output",
        type=Path,
        required=True,
        help="Output file"
    )
    
    args = parser.parse_args()
    
    # Validate all inputs exist
    for f in args.files:
        if not f.exists():
            parser.error(f"File not found: {f}")
    
    # Concatenate
    content = ""
    for f in args.files:
        content += f.read_text()
    
    args.output.write_text(content)
    print(f"Wrote {len(args.files)} files to {args.output}")

if __name__ == "__main__":
    main()
```

### Scenario 4: Configuration from Arguments

Building a config object from CLI args:

```python
#!/usr/bin/env python3
"""Application with structured config."""

import argparse
from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    input_path: Path
    output_path: Path
    batch_size: int
    verbose: bool
    dry_run: bool
    log_level: str

def parse_config() -> Config:
    parser = argparse.ArgumentParser()
    
    # Required
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    
    # Optional with defaults
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=100,
        help="Batch size (default: %(default)s)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without doing it"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO"
    )
    
    args = parser.parse_args()
    
    # Convert to typed config
    return Config(
        input_path=args.input,
        output_path=args.output,
        batch_size=args.batch_size,
        verbose=args.verbose,
        dry_run=args.dry_run,
        log_level=args.log_level,
    )

def main():
    config = parse_config()
    
    if config.verbose:
        print(f"Config: {config}")
    
    if config.dry_run:
        print("DRY RUN: Would process files")
        return
    
    # ... actual work ...

if __name__ == "__main__":
    main()
```

### Scenario 5: Custom Type Validation

For arguments that need special validation:

```python
#!/usr/bin/env python3
"""Script with validated arguments."""

import argparse
from pathlib import Path
from datetime import date

def valid_date(s: str) -> date:
    """Parse date in YYYY-MM-DD format."""
    try:
        return date.fromisoformat(s)
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Invalid date format: {s}. Use YYYY-MM-DD"
        )

def existing_file(s: str) -> Path:
    """Validate that file exists."""
    path = Path(s)
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File not found: {s}")
    return path

def positive_int(s: str) -> int:
    """Parse positive integer."""
    try:
        value = int(s)
        if value <= 0:
            raise ValueError()
        return value
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"Must be a positive integer: {s}"
        )

def main():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "-d", "--date",
        type=valid_date,
        default=date.today(),
        help="Date in YYYY-MM-DD format"
    )
    
    parser.add_argument(
        "-f", "--file",
        type=existing_file,
        help="Path to existing file"
    )
    
    parser.add_argument(
        "-n", "--count",
        type=positive_int,
        default=10,
        help="Positive integer count"
    )
    
    args = parser.parse_args()
    print(f"Date: {args.date}")
    print(f"File: {args.file}")
    print(f"Count: {args.count}")

if __name__ == "__main__":
    main()
```

**Error messages:**

```bash
$ python script.py --date 2024-13-01
error: argument -d/--date: Invalid date format: 2024-13-01. Use YYYY-MM-DD

$ python script.py --count -5
error: argument -n/--count: Must be a positive integer: -5
```

### Production Patterns

#### Pattern 1: Combining with environment variables

```python
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    
    # CLI takes precedence over environment
    parser.add_argument(
        "--api-key",
        default=os.environ.get("API_KEY"),
        help="API key (or set API_KEY env var)"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.environ.get("DEBUG", "").lower() == "true",
        help="Enable debug mode (or set DEBUG=true)"
    )
    
    args = parser.parse_args()
    
    if not args.api_key:
        parser.error("API key required: use --api-key or set API_KEY")
```

#### Pattern 2: Argument groups for organization

```python
parser = argparse.ArgumentParser()

# Group related arguments
input_group = parser.add_argument_group("Input options")
input_group.add_argument("-i", "--input", required=True)
input_group.add_argument("-f", "--format", default="csv")

output_group = parser.add_argument_group("Output options")
output_group.add_argument("-o", "--output", required=True)
output_group.add_argument("--compress", action="store_true")

debug_group = parser.add_argument_group("Debug options")
debug_group.add_argument("-v", "--verbose", action="store_true")
debug_group.add_argument("--dry-run", action="store_true")
```

#### Pattern 3: Mutually exclusive options

```python
parser = argparse.ArgumentParser()

# Only one of these can be specified
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument("-f", "--file", help="Read from file")
group.add_argument("-u", "--url", help="Read from URL")
group.add_argument("-s", "--stdin", action="store_true", help="Read from stdin")

# Usage:
# script.py -f data.txt   ✓
# script.py -u http://... ✓
# script.py -f x -u y     ✗ Error: mutually exclusive
```

### What Breaks: Common Mistakes

**1. Wrong nargs for lists**

```python
# Wrong: Only takes one value
parser.add_argument("--files")

# Right: Takes multiple values
parser.add_argument("--files", nargs="+")
```

**2. Positional arguments with defaults**

```python
# Doesn't work as expected
parser.add_argument("input", default="file.txt")  # Still required!

# Fix: Use nargs='?'
parser.add_argument("input", nargs='?', default="file.txt")
```

**3. Boolean arguments**

```python
# Wrong: type=bool doesn't work
parser.add_argument("--flag", type=bool)

# Right: Use action
parser.add_argument("--flag", action="store_true")

# Or for explicit true/false:
parser.add_argument(
    "--flag",
    type=lambda x: x.lower() in ('true', '1', 'yes'),
    default=False
)
```

**4. Forgetting to call parse_args()**

```python
parser = argparse.ArgumentParser()
parser.add_argument("--name")
# ... forgot to parse!
print(args.name)  # NameError: args not defined

# Fix:
args = parser.parse_args()
print(args.name)
```

---

## Summary: The Mental Checklist

1. **Positional or optional?**
   - Essential, always needed → Positional
   - Modifier, has default → Optional with `--`

2. **What action?**
   - Takes a value → Default (`store`)
   - Boolean flag → `store_true` or `store_false`
   - Collect multiple → `append` or `nargs="+"`

3. **What type?**
   - String is default
   - Use `type=int`, `type=float`, `type=Path`
   - Custom validator? Write a function that raises `ArgumentTypeError`

4. **Required or optional?**
   - Positionals are always required (unless `nargs='?'`)
   - Optionals default to not required (use `required=True` to force)

5. **Good help text?**
   - Use `help=` on every argument
   - Use `%(default)s` to show defaults
   - Add examples in `epilog=`
