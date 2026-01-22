# Data Cleaning

Data cleaning is the systematic transformation of messy data into a consistent, usable form. It is not ad-hoc fixes — it is a structured process with clear principles.

This guide covers cleaning for business data: mixed types, real-world formats, the mess that comes from CRM systems, spreadsheets, and manual entry. We use polars for execution, but the principles apply to any tool.

---

## 1. The Cleaning Hierarchy

Cleaning operations have dependencies. The order matters.

### Order of Operations

Clean in this sequence:

1. **Type coercion** — Get the right data types first
2. **Structural cleaning** — Fix format issues within each type
3. **Value normalization** — Standardize valid values
4. **Outlier/anomaly handling** — Address extreme values
5. **Missing value treatment** — Handle nulls last

Why this order?

- You cannot clean numeric formatting if the column is still a string
- You cannot detect outliers if values are not yet numeric
- You cannot impute missing values if you haven't defined what "valid" looks like

### Type Coercion Before Value Cleaning

A column with values like `"$1,234.56"`, `"N/A"`, `"1000"` must be:

1. Recognized as "intended to be numeric"
2. Have formatting stripped (`$`, `,`)
3. Have invalid values handled (`N/A` → null)
4. Cast to a numeric type
5. Then checked for outliers, imputed, etc.

Attempting value cleaning before type coercion leads to errors:

```python
# Wrong order — fails because column is still string
df.with_columns(
    pl.col("amount").cast(pl.Float64)  # Fails on "$1,234.56"
)

# Correct order
df.with_columns(
    pl.col("amount")
    .str.replace_all(r"[$,]", "")      # Remove formatting
    .str.replace("N/A", "")            # Handle placeholders
    .cast(pl.Float64, strict=False)    # Cast with null for failures
    .alias("amount_clean")
)
```

### Cleaning as Transformation, Not Mutation

Never overwrite original data during cleaning. Add new columns:

```python
# Bad: overwrites original
df = df.with_columns(pl.col("amount").str.replace_all(r"[$,]", ""))

# Good: preserves original
df = df.with_columns(
    pl.col("amount")
    .str.replace_all(r"[$,]", "")
    .alias("amount_clean")
)
```

Benefits:
- Debug by comparing original to cleaned
- Rollback if cleaning logic is wrong
- Audit trail of transformations
- Different downstream uses may need different cleaning

---

## 2. Type Coercion

Type coercion converts data to the correct type. It's the foundation of all other cleaning.

### Strings to Numbers

Numeric data often arrives as strings with formatting:

```python
# Common numeric string formats
# "$1,234.56"  → currency
# "1 234,56"   → European format
# "(100.00)"   → accounting negative
# "45%"        → percentage
# "N/A"        → placeholder

def clean_numeric_string(df: pl.DataFrame, col: str, new_col: str = None) -> pl.DataFrame:
    """Convert formatted string to numeric."""
    if new_col is None:
        new_col = f"{col}_clean"
    
    return df.with_columns(
        pl.col(col)
        # Remove currency symbols and thousands separators
        .str.replace_all(r"[$€£¥,\s]", "")
        # Handle percentage (divide by 100 later if needed)
        .str.replace("%", "")
        # Handle accounting negatives: (100) → -100
        .str.replace(r"^\((.+)\)$", "-$1")
        # Replace common null representations
        .str.replace(r"^(N/A|NA|null|NULL|None|-|—)$", "")
        # Cast to float
        .cast(pl.Float64, strict=False)
        .alias(new_col)
    )
```

**European vs US formats:** In Europe, `1.234,56` means 1234.56. Detect and handle:

```python
def detect_numeric_format(sample: list) -> str:
    """Detect European vs US numeric format."""
    # European: comma as decimal, period or space as thousands
    # US: period as decimal, comma as thousands
    
    for val in sample:
        if isinstance(val, str):
            # Pattern: digits, period, 3 digits, comma = European
            if re.match(r"\d{1,3}\.\d{3},\d", val):
                return "european"
            # Pattern: digits, comma, 3 digits, period = US
            if re.match(r"\d{1,3},\d{3}\.\d", val):
                return "us"
    
    return "unknown"

def normalize_european_numeric(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Convert European format to standard."""
    return df.with_columns(
        pl.col(col)
        .str.replace_all(r"\.", "")     # Remove thousands separator
        .str.replace(",", ".")          # Convert decimal comma to period
        .cast(pl.Float64, strict=False)
        .alias(f"{col}_clean")
    )
```

### Strings to Dates

Date parsing is notoriously error-prone:

```python
# Common date formats in the wild
# "2024-01-15"      → ISO (unambiguous)
# "01/15/2024"      → US (MM/DD/YYYY)
# "15/01/2024"      → European (DD/MM/YYYY)
# "Jan 15, 2024"    → Natural language
# "15-Jan-24"       → Mixed
# "20240115"        → Compact

def parse_date_column(
    df: pl.DataFrame,
    col: str,
    format: str = None,
    new_col: str = None
) -> pl.DataFrame:
    """Parse string column to date with format handling."""
    if new_col is None:
        new_col = f"{col}_date"
    
    if format:
        # Explicit format provided
        return df.with_columns(
            pl.col(col).str.to_date(format, strict=False).alias(new_col)
        )
    
    # Try common formats
    return df.with_columns(
        pl.coalesce(
            pl.col(col).str.to_date("%Y-%m-%d", strict=False),     # ISO
            pl.col(col).str.to_date("%m/%d/%Y", strict=False),     # US
            pl.col(col).str.to_date("%d/%m/%Y", strict=False),     # EU
            pl.col(col).str.to_date("%Y%m%d", strict=False),       # Compact
            pl.col(col).str.to_date("%b %d, %Y", strict=False),    # Natural
        ).alias(new_col)
    )
```

**Ambiguous dates:** `01/02/2024` is January 2nd (US) or February 1st (EU). When in doubt:

```python
def detect_date_format(df: pl.DataFrame, col: str) -> str:
    """Detect likely date format by analyzing values."""
    # Sample values that have day > 12 (unambiguous)
    sample = df.filter(pl.col(col).is_not_null()).select(col).head(1000)
    
    for val in sample[col].to_list():
        if isinstance(val, str):
            parts = re.split(r"[/\-.]", val)
            if len(parts) >= 2:
                try:
                    # If first part > 12, it's a day (EU format)
                    if int(parts[0]) > 12:
                        return "DD/MM/YYYY"
                    # If second part > 12, it's a day (US format)
                    if int(parts[1]) > 12:
                        return "MM/DD/YYYY"
                except ValueError:
                    pass
    
    return "AMBIGUOUS"
```

### Casting to Categoricals

String columns with limited unique values should be categorical:

```python
def optimize_string_to_categorical(
    df: pl.DataFrame,
    max_cardinality: int = 100
) -> pl.DataFrame:
    """Convert low-cardinality string columns to categorical."""
    for col in df.columns:
        if df.schema[col] == pl.Utf8:
            unique_count = df.select(pl.col(col).n_unique()).item()
            if unique_count <= max_cardinality:
                df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df
```

### Handling Coercion Failures

When casting fails, you need to know what failed:

```python
def cast_with_failure_tracking(
    df: pl.DataFrame,
    col: str,
    target_type: pl.DataType
) -> pl.DataFrame:
    """Cast column, tracking which rows failed."""
    return df.with_columns(
        # Attempt cast
        pl.col(col).cast(target_type, strict=False).alias(f"{col}_clean"),
        # Track failures (original not null but result is null)
        (
            pl.col(col).is_not_null() &
            pl.col(col).cast(target_type, strict=False).is_null()
        ).alias(f"{col}_cast_failed")
    )

# Check what failed
df = cast_with_failure_tracking(df, "amount", pl.Float64)
failures = df.filter(pl.col("amount_cast_failed"))
print(f"Failed to cast {failures.height} values:")
print(failures.select("amount").unique())
```

---

## 3. Numeric Cleaning

Once data is numeric, clean the values themselves.

### Removing Formatting Artifacts

Sometimes formatting persists after initial coercion:

```python
def clean_numeric_column(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Clean a numeric column of common issues."""
    return df.with_columns(
        pl.when(pl.col(col).is_infinite())
        .then(None)  # Replace infinity with null
        .when(pl.col(col).is_nan())
        .then(None)  # Replace NaN with null
        .otherwise(pl.col(col))
        .alias(col)
    )
```

### Handling Negative Formats

Accounting systems often show negatives as `(100)` instead of `-100`:

```python
# If still in string format
df = df.with_columns(
    pl.when(pl.col("amount").str.contains(r"^\("))
    .then(
        pl.lit("-") + pl.col("amount").str.replace_all(r"[()]", "")
    )
    .otherwise(pl.col("amount"))
    .alias("amount_normalized")
)
```

### Precision and Rounding

Floating-point precision can cause issues:

```python
# 0.1 + 0.2 = 0.30000000000000004 in floating point

def round_currency(df: pl.DataFrame, col: str, decimals: int = 2) -> pl.DataFrame:
    """Round currency values to avoid floating-point artifacts."""
    return df.with_columns(
        pl.col(col).round(decimals).alias(col)
    )

# Or use fixed-point representation for currency
df = df.with_columns(
    (pl.col("amount") * 100).round(0).cast(pl.Int64).alias("amount_cents")
)
```

### Outlier Treatment

Decide what to do with outliers based on context:

```python
def handle_outliers(
    df: pl.DataFrame,
    col: str,
    method: str = "cap",  # "cap", "remove", "flag"
    lower_pct: float = 0.01,
    upper_pct: float = 0.99
) -> pl.DataFrame:
    """Handle outliers using specified method."""
    
    lower = df.select(pl.col(col).quantile(lower_pct)).item()
    upper = df.select(pl.col(col).quantile(upper_pct)).item()
    
    if method == "cap":
        # Winsorize: cap at percentile bounds
        return df.with_columns(
            pl.col(col).clip(lower, upper).alias(f"{col}_capped")
        )
    
    elif method == "remove":
        # Filter out outliers
        return df.filter(
            (pl.col(col) >= lower) & (pl.col(col) <= upper)
        )
    
    elif method == "flag":
        # Add outlier flag
        return df.with_columns(
            (
                (pl.col(col) < lower) | (pl.col(col) > upper)
            ).alias(f"{col}_is_outlier")
        )
    
    return df
```

---

## 4. Text Cleaning

Text requires specialized cleaning that respects semantic meaning.

### Case Normalization Decisions

Not all text should be lowercased:

```python
def normalize_case(df: pl.DataFrame, col: str, strategy: str = "lower") -> pl.DataFrame:
    """Normalize case with different strategies."""
    
    if strategy == "lower":
        return df.with_columns(pl.col(col).str.to_lowercase().alias(col))
    
    elif strategy == "upper":
        return df.with_columns(pl.col(col).str.to_uppercase().alias(col))
    
    elif strategy == "title":
        # Capitalize first letter of each word
        return df.with_columns(pl.col(col).str.to_titlecase().alias(col))
    
    elif strategy == "preserve":
        # Don't change case
        return df
    
    return df

# Different columns need different strategies
df = df.with_columns(
    pl.col("email").str.to_lowercase().alias("email"),           # Always lowercase
    pl.col("customer_name").str.to_titlecase().alias("customer_name"),  # Title case
    pl.col("product_code").str.to_uppercase().alias("product_code"),     # Always uppercase
)
```

### Whitespace and Control Characters

```python
def clean_whitespace(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Comprehensive whitespace cleaning."""
    return df.with_columns(
        pl.col(col)
        # Remove control characters
        .str.replace_all(r"[\x00-\x1F\x7F]", "")
        # Normalize various whitespace to regular space
        .str.replace_all(r"[\t\r\n\u00A0\u2003]", " ")
        # Collapse multiple spaces
        .str.replace_all(r" +", " ")
        # Trim edges
        .str.strip_chars()
        .alias(col)
    )
```

### Name Standardization

Personal names have many edge cases:

```python
def standardize_name(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Standardize personal names."""
    return df.with_columns(
        pl.col(col)
        # Clean whitespace first
        .str.strip_chars()
        .str.replace_all(r"\s+", " ")
        # Handle common title/suffix patterns
        .str.replace(r"^(Mr\.?|Mrs\.?|Ms\.?|Dr\.?)\s+", "")
        .str.replace(r"\s+(Jr\.?|Sr\.?|III?|IV)$", "")
        # Title case
        .str.to_titlecase()
        # Handle McDonald, O'Brien, etc. (imperfect but common)
        .str.replace(r"\bMc([a-z])", "Mc$1")  # McDonald
        .str.replace(r"\bO'([a-z])", "O'$1")  # O'Brien
        .alias(f"{col}_clean")
    )
```

### Email Normalization

```python
def normalize_email(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Normalize email addresses."""
    return df.with_columns(
        pl.col(col)
        # Lowercase (email local parts are case-insensitive in practice)
        .str.to_lowercase()
        # Strip whitespace
        .str.strip_chars()
        # Remove leading/trailing dots
        .str.replace(r"^\.+|\.+$", "")
        .alias(f"{col}_clean")
    )

def validate_email_format(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Flag invalid email formats."""
    email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
    
    return df.with_columns(
        pl.col(col).str.contains(email_pattern).alias(f"{col}_valid")
    )
```

### Phone Number Normalization

```python
def normalize_phone(df: pl.DataFrame, col: str, country: str = "US") -> pl.DataFrame:
    """Normalize phone numbers to consistent format."""
    return df.with_columns(
        pl.col(col)
        # Remove all non-digit characters
        .str.replace_all(r"[^\d]", "")
        # Handle US: add country code if 10 digits
        .map_elements(
            lambda x: f"+1{x}" if len(x) == 10 else f"+{x}" if len(x) > 10 else x,
            return_dtype=pl.Utf8
        )
        .alias(f"{col}_clean")
    )
```

---

## 5. Date Cleaning

Dates have unique challenges: time zones, invalid values, and business logic.

### Format Standardization

Convert all dates to ISO format internally:

```python
def standardize_dates(df: pl.DataFrame, date_cols: list) -> pl.DataFrame:
    """Standardize all date columns to consistent format."""
    for col in date_cols:
        if df.schema[col] == pl.Utf8:
            # Parse string to date
            df = df.with_columns(
                pl.col(col).str.to_date("%Y-%m-%d", strict=False).alias(col)
            )
    return df
```

### Timezone Handling

```python
from datetime import timezone

def normalize_timezone(
    df: pl.DataFrame,
    col: str,
    source_tz: str = "UTC",
    target_tz: str = "UTC"
) -> pl.DataFrame:
    """Convert datetime column between timezones."""
    return df.with_columns(
        pl.col(col)
        .dt.replace_time_zone(source_tz)
        .dt.convert_time_zone(target_tz)
        .alias(col)
    )
```

### Invalid Date Repair

Some "dates" are clearly wrong:

```python
from datetime import date

def repair_invalid_dates(
    df: pl.DataFrame,
    col: str,
    min_date: date = date(1900, 1, 1),
    max_date: date = None
) -> pl.DataFrame:
    """Flag or repair invalid dates."""
    if max_date is None:
        max_date = date.today()
    
    return df.with_columns(
        pl.when(
            (pl.col(col) < min_date) | (pl.col(col) > max_date)
        )
        .then(None)  # Replace invalid with null
        .otherwise(pl.col(col))
        .alias(f"{col}_clean"),
        
        # Flag what was invalid
        (
            (pl.col(col) < min_date) | (pl.col(col) > max_date)
        ).alias(f"{col}_was_invalid")
    )
```

### Business Logic Date Fixes

Sometimes dates need business context:

```python
def fix_future_dates(df: pl.DataFrame, col: str) -> pl.DataFrame:
    """Handle future dates based on likely cause."""
    today = date.today()
    
    return df.with_columns(
        pl.when(pl.col(col) > today)
        .then(
            # Assume year typo: 2025 → 2024 if date is in next year
            pl.when(pl.col(col).dt.year() == today.year + 1)
            .then(pl.col(col).dt.offset_by("-1y"))
            .otherwise(None)  # Can't fix, set null
        )
        .otherwise(pl.col(col))
        .alias(f"{col}_fixed")
    )
```

---

## 6. Categorical Cleaning

Categorical data needs value standardization.

### Value Standardization

Map variations to canonical values:

```python
def standardize_categories(
    df: pl.DataFrame,
    col: str,
    mapping: dict
) -> pl.DataFrame:
    """Map category variations to standard values."""
    # Build case-insensitive mapping
    lower_mapping = {k.lower(): v for k, v in mapping.items()}
    
    return df.with_columns(
        pl.col(col)
        .str.to_lowercase()
        .str.strip_chars()
        .replace(lower_mapping)
        .alias(f"{col}_clean")
    )

# Example: status standardization
status_mapping = {
    "yes": "Y", "y": "Y", "true": "Y", "1": "Y",
    "no": "N", "n": "N", "false": "N", "0": "N",
    "pending": "P", "pend": "P", "in progress": "P",
}

df = standardize_categories(df, "status", status_mapping)
```

### Typo Correction

Use fuzzy matching for typo correction:

```python
from difflib import get_close_matches

def fix_categorical_typos(
    df: pl.DataFrame,
    col: str,
    valid_values: list,
    threshold: float = 0.8
) -> pl.DataFrame:
    """Fix typos by matching to closest valid value."""
    
    def find_closest(value):
        if value is None or value in valid_values:
            return value
        matches = get_close_matches(value.lower(), [v.lower() for v in valid_values], n=1, cutoff=threshold)
        if matches:
            # Return the original case version
            idx = [v.lower() for v in valid_values].index(matches[0])
            return valid_values[idx]
        return value  # No match, keep original
    
    return df.with_columns(
        pl.col(col)
        .map_elements(find_closest, return_dtype=pl.Utf8)
        .alias(f"{col}_fixed")
    )

# Example
valid_statuses = ["Pending", "Shipped", "Delivered", "Cancelled"]
df = fix_categorical_typos(df, "status", valid_statuses)
# "Shiped" → "Shipped", "Deliverd" → "Delivered"
```

### Unknown/Other Category Handling

Handle values that don't match any known category:

```python
def handle_unknown_categories(
    df: pl.DataFrame,
    col: str,
    valid_values: list,
    unknown_label: str = "Other"
) -> pl.DataFrame:
    """Replace unknown categories with a standard label."""
    return df.with_columns(
        pl.when(pl.col(col).is_in(valid_values))
        .then(pl.col(col))
        .otherwise(pl.lit(unknown_label))
        .alias(f"{col}_clean")
    )
```

---

## 7. Preserving Lineage

Tracking what changed, when, and why is essential for debugging and auditing.

### Original vs Cleaned Columns

Keep both versions:

```python
def clean_with_lineage(df: pl.DataFrame, col: str, cleaning_func) -> pl.DataFrame:
    """Apply cleaning while preserving original."""
    return df.with_columns(
        pl.col(col).alias(f"{col}_original"),
        cleaning_func(pl.col(col)).alias(col),
    )
```

### Change Logs

Track what changed:

```python
def create_change_log(
    df: pl.DataFrame,
    original_col: str,
    cleaned_col: str
) -> pl.DataFrame:
    """Create a log of rows where values changed."""
    return df.with_columns(
        (pl.col(original_col) != pl.col(cleaned_col)).alias("was_changed"),
    ).filter(pl.col("was_changed")).select([
        original_col,
        cleaned_col,
        "was_changed"
    ])

# Usage
changes = create_change_log(df, "status_original", "status")
print(f"Changed {changes.height} values")
changes.write_csv("cleaning_changes.csv")
```

### Reversibility

Design cleaning to be reversible:

```python
def reversible_clean(df: pl.DataFrame, operations: list) -> tuple:
    """Apply cleaning operations, return cleaned df and reversal info."""
    original = df.clone()
    cleaned = df
    
    for op in operations:
        cleaned = op(cleaned)
    
    # Store what was changed for potential rollback
    diff = original.join(cleaned, on="id", suffix="_cleaned")
    
    return cleaned, diff
```

---

## Summary

Data cleaning is structured transformation, not ad-hoc fixes.

**Order matters:** Type coercion → structural cleaning → value normalization → outlier handling → missing values.

**Preserve originals:** Never mutate in place. Keep original columns alongside cleaned versions.

**Type-specific strategies:** Numbers, text, dates, and categoricals each have unique cleaning patterns.

**Track lineage:** Know what changed, when, and why. Make cleaning reversible.

---

## Interview Framing

**"How do you approach cleaning a messy dataset?"**

"I follow a strict order. First, type coercion — getting strings to numbers, dates parsed correctly. You can't clean values until types are right. Then structural cleaning: whitespace, formatting artifacts. Then value normalization: standardizing categories, fixing typos. Then outlier handling and finally missing values. I always preserve original data in separate columns so I can debug and audit changes."

**"What's the trickiest part of data cleaning?"**

"Type coercion, especially dates. Is '01/02/2024' January 2nd or February 1st? You need to analyze the data to detect the format, not guess. For numbers, European versus US formatting — commas and periods swap meaning. And 'missing' values: null, empty string, 'N/A', '-' are all different representations of the same concept. You have to normalize all of them before real cleaning can start."
