# Data Profiling

Data profiling is the systematic examination of data to understand its structure, content, and quality before any transformation. It answers the question: "What do I actually have?"

Skipping profiling is the most common mistake in data engineering. Teams jump straight to cleaning, make assumptions about the data, and discover problems in production. Profiling costs minutes upfront and saves hours of debugging later.

This guide uses polars with business data examples: mixed types, real-world messiness, the kind of data that comes from CRM exports, ERP systems, and spreadsheets.

---

## 1. Why Profile Before Anything Else

### The "Look Before You Leap" Principle

When you receive a dataset, you know nothing about it. The column names suggest meaning, but:

- `amount` might be stored as a string with currency symbols
- `date` might have three different formats
- `status` might have 47 unique values when you expected 5
- `customer_id` might have duplicates when it should be unique

Profiling reveals the actual state of the data, not the intended state.

### What Profiling Reveals That head() Doesn't

Looking at the first few rows is not profiling. The first rows are often the cleanest — they were entered first, tested manually, or simply don't represent the full distribution.

```python
import polars as pl

df = pl.read_csv("orders.csv")
df.head(5)  # Shows 5 clean rows
```

What `head()` misses:
- **Nulls in row 50,000** — you won't see them in the first 5 rows
- **Type inconsistency** — row 1 has "100.00", row 10,000 has "N/A"
- **Distribution skew** — most orders are $10-50, but row 75,000 has $999,999
- **Encoding issues** — row 30,000 has mojibake in the customer name

Profiling examines the entire dataset statistically. It finds the problems that hide in the tail.

### Cost of Skipping Profiling

What happens when you skip profiling:

**Silent failures:** You write a cleaning pipeline assuming `amount` is numeric. It works on your test data. In production, it fails on "N/A" values you never saw.

**Wrong assumptions:** You assume `order_date` is always in the past. Six months later, you discover future dates that broke your time-series analysis.

**Wasted effort:** You spend hours debugging why aggregations are wrong. The cause: duplicate `order_id` values you assumed were unique.

**Production incidents:** Your pipeline runs for months. One day, a source system change introduces new status codes. Your downstream systems break.

Profiling is cheap insurance against all of these.

---

## 2. Schema Profiling

Schema profiling examines the structure of data: what columns exist, what types they have, and whether those types match reality.

### Column Types: Inferred vs Actual

When you load data, the reader infers types:

```python
df = pl.read_csv("orders.csv")
print(df.schema)

# Output:
# {'order_id': Int64, 'customer_name': Utf8, 'order_amount': Utf8, 
#  'order_date': Utf8, 'status': Utf8}
```

Notice: `order_amount` is `Utf8` (string), not a number. The reader saw something that wasn't purely numeric — maybe "$100.00" or "N/A" — and inferred string.

Compare inferred types against expected types:

```python
expected_schema = {
    "order_id": pl.Int64,
    "customer_name": pl.Utf8,
    "order_amount": pl.Float64,  # Expected numeric
    "order_date": pl.Date,       # Expected date
    "status": pl.Utf8,
}

def compare_schemas(df: pl.DataFrame, expected: dict) -> dict:
    """Compare actual schema against expected."""
    actual = df.schema
    mismatches = {}
    
    for col, expected_type in expected.items():
        if col not in actual:
            mismatches[col] = {"expected": expected_type, "actual": "MISSING"}
        elif actual[col] != expected_type:
            mismatches[col] = {"expected": expected_type, "actual": actual[col]}
    
    # Check for unexpected columns
    for col in actual:
        if col not in expected:
            mismatches[col] = {"expected": "NOT EXPECTED", "actual": actual[col]}
    
    return mismatches

mismatches = compare_schemas(df, expected_schema)
print(f"Schema mismatches: {mismatches}")
```

### Type Mismatches: Numbers Stored as Strings

The most common mismatch: numeric data stored as strings. This happens because:

- Currency symbols: "$1,234.56"
- Percentage signs: "45%"
- Placeholder values: "N/A", "-", "TBD"
- Leading zeros preserved: "00123"

Detect these by checking if string columns are "mostly numeric":

```python
def detect_numeric_strings(df: pl.DataFrame) -> dict:
    """Find string columns that contain mostly numeric values."""
    results = {}
    
    for col in df.columns:
        if df.schema[col] == pl.Utf8:
            # Try to cast to numeric, count successes
            numeric_count = df.select(
                pl.col(col).str.replace_all(r"[$,%\s]", "")
                .str.strip_chars()
                .cast(pl.Float64, strict=False)
                .is_not_null()
                .sum()
            ).item()
            
            total = df.height
            non_null = df.select(pl.col(col).is_not_null().sum()).item()
            
            if non_null > 0:
                numeric_ratio = numeric_count / non_null
                if numeric_ratio > 0.9:  # 90%+ numeric
                    results[col] = {
                        "numeric_ratio": numeric_ratio,
                        "recommendation": "Consider casting to numeric"
                    }
    
    return results
```

### Schema Drift Detection

Schema drift is when the structure changes over time. Yesterday's file had 10 columns; today's has 11. Last week `status` had 5 values; this week it has 6.

For ongoing pipelines, track schema and alert on changes:

```python
import json
from datetime import datetime

def capture_schema_snapshot(df: pl.DataFrame) -> dict:
    """Capture schema for drift detection."""
    return {
        "captured_at": datetime.now().isoformat(),
        "columns": list(df.columns),
        "types": {col: str(dtype) for col, dtype in df.schema.items()},
        "row_count": df.height,
    }

def detect_drift(current: dict, baseline: dict) -> list:
    """Compare current schema against baseline."""
    issues = []
    
    # New columns
    new_cols = set(current["columns"]) - set(baseline["columns"])
    if new_cols:
        issues.append(f"New columns: {new_cols}")
    
    # Removed columns
    removed_cols = set(baseline["columns"]) - set(current["columns"])
    if removed_cols:
        issues.append(f"Removed columns: {removed_cols}")
    
    # Type changes
    for col in set(current["columns"]) & set(baseline["columns"]):
        if current["types"][col] != baseline["types"][col]:
            issues.append(
                f"Type change in {col}: {baseline['types'][col]} → {current['types'][col]}"
            )
    
    return issues
```

---

## 3. Univariate Profiling

Univariate profiling examines each column independently. Different types require different profiles.

### Numeric Profiling

For numeric columns, profile the distribution:

```python
def profile_numeric(df: pl.DataFrame, col: str) -> dict:
    """Profile a numeric column."""
    return df.select(
        # Basic stats
        pl.col(col).count().alias("count"),
        pl.col(col).null_count().alias("null_count"),
        pl.col(col).n_unique().alias("unique_count"),
        
        # Distribution
        pl.col(col).min().alias("min"),
        pl.col(col).max().alias("max"),
        pl.col(col).mean().alias("mean"),
        pl.col(col).median().alias("median"),
        pl.col(col).std().alias("std"),
        
        # Percentiles
        pl.col(col).quantile(0.01).alias("p01"),
        pl.col(col).quantile(0.05).alias("p05"),
        pl.col(col).quantile(0.25).alias("p25"),
        pl.col(col).quantile(0.75).alias("p75"),
        pl.col(col).quantile(0.95).alias("p95"),
        pl.col(col).quantile(0.99).alias("p99"),
        
        # Quality indicators
        (pl.col(col) < 0).sum().alias("negative_count"),
        (pl.col(col) == 0).sum().alias("zero_count"),
    ).to_dicts()[0]

# Usage
profile = profile_numeric(df, "order_amount")
print(f"Range: {profile['min']} to {profile['max']}")
print(f"Mean: {profile['mean']:.2f}, Median: {profile['median']:.2f}")
```

**Outlier detection:** Compare p01/p99 to min/max. Large gaps indicate outliers.

```python
def detect_outliers_iqr(df: pl.DataFrame, col: str, multiplier: float = 1.5) -> dict:
    """Detect outliers using IQR method."""
    stats = df.select(
        pl.col(col).quantile(0.25).alias("q1"),
        pl.col(col).quantile(0.75).alias("q3"),
    ).to_dicts()[0]
    
    iqr = stats["q3"] - stats["q1"]
    lower_bound = stats["q1"] - multiplier * iqr
    upper_bound = stats["q3"] + multiplier * iqr
    
    outliers = df.filter(
        (pl.col(col) < lower_bound) | (pl.col(col) > upper_bound)
    )
    
    return {
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "outlier_count": outliers.height,
        "outlier_pct": outliers.height / df.height * 100,
    }
```

### Text Profiling

For text columns, profile length and content patterns:

```python
def profile_text(df: pl.DataFrame, col: str) -> dict:
    """Profile a text column."""
    text = pl.col(col)
    
    return df.select(
        # Basic stats
        text.count().alias("count"),
        text.null_count().alias("null_count"),
        text.n_unique().alias("unique_count"),
        
        # Length distribution
        text.str.len_chars().min().alias("min_length"),
        text.str.len_chars().max().alias("max_length"),
        text.str.len_chars().mean().alias("mean_length"),
        text.str.len_chars().median().alias("median_length"),
        
        # Empty strings
        (text == "").sum().alias("empty_count"),
        
        # Whitespace issues
        (text != text.str.strip_chars()).sum().alias("has_leading_trailing_space"),
        
        # Pattern checks
        text.str.contains(r"^\s*$").sum().alias("whitespace_only_count"),
        text.str.contains(r"[^\x00-\x7F]").sum().alias("non_ascii_count"),
    ).to_dicts()[0]
```

**Top values** help understand categorical-like text:

```python
def top_values(df: pl.DataFrame, col: str, n: int = 10) -> pl.DataFrame:
    """Get most frequent values."""
    return (
        df.group_by(col)
        .agg(pl.count().alias("frequency"))
        .sort("frequency", descending=True)
        .head(n)
        .with_columns(
            (pl.col("frequency") / df.height * 100).alias("pct")
        )
    )
```

### Date Profiling

For date columns, profile the range and detect anomalies:

```python
from datetime import date, datetime

def profile_date(df: pl.DataFrame, col: str) -> dict:
    """Profile a date column."""
    today = date.today()
    
    return df.select(
        # Basic stats
        pl.col(col).count().alias("count"),
        pl.col(col).null_count().alias("null_count"),
        pl.col(col).n_unique().alias("unique_count"),
        
        # Range
        pl.col(col).min().alias("min_date"),
        pl.col(col).max().alias("max_date"),
        
        # Anomalies
        (pl.col(col) > today).sum().alias("future_dates"),
        (pl.col(col) < date(1900, 1, 1)).sum().alias("very_old_dates"),
        
        # Gaps (days between min and max)
        (pl.col(col).max() - pl.col(col).min()).dt.total_days().alias("date_range_days"),
    ).to_dicts()[0]
```

### Categorical Profiling

For categorical columns, profile cardinality and distribution:

```python
def profile_categorical(df: pl.DataFrame, col: str) -> dict:
    """Profile a categorical column."""
    value_counts = df.group_by(col).agg(pl.count().alias("n"))
    
    return {
        "unique_count": value_counts.height,
        "null_count": df.select(pl.col(col).null_count()).item(),
        "top_value": value_counts.sort("n", descending=True).head(1)[col].item(),
        "top_value_count": value_counts.sort("n", descending=True).head(1)["n"].item(),
        "singleton_count": value_counts.filter(pl.col("n") == 1).height,
        "values": value_counts.sort("n", descending=True).head(20).to_dicts(),
    }
```

**High cardinality warning:** If a "categorical" column has unique values approaching the row count, it's not really categorical.

```python
def check_cardinality(df: pl.DataFrame, col: str, threshold: float = 0.5) -> str:
    """Check if cardinality is suspiciously high."""
    unique = df.select(pl.col(col).n_unique()).item()
    total = df.height
    ratio = unique / total
    
    if ratio > threshold:
        return f"WARNING: {col} has {ratio:.1%} unique values - may not be categorical"
    return f"OK: {col} has {unique} unique values ({ratio:.1%})"
```

---

## 4. Missing Value Analysis

Missing values are not just "nulls." They have patterns that reveal data quality issues.

### Null Patterns: Random vs Systematic

**Random missing:** Values are missing independently, scattered throughout the data. Usually not a big problem — imputation or removal works.

**Systematic missing:** Values are missing for a reason. All orders from vendor X have no shipping date. All customers from region Y have no phone number. This reveals data collection issues.

Detect patterns:

```python
def missing_value_report(df: pl.DataFrame) -> pl.DataFrame:
    """Generate missing value report for all columns."""
    reports = []
    
    for col in df.columns:
        null_count = df.select(pl.col(col).null_count()).item()
        reports.append({
            "column": col,
            "null_count": null_count,
            "null_pct": null_count / df.height * 100,
            "dtype": str(df.schema[col]),
        })
    
    return pl.DataFrame(reports).sort("null_pct", descending=True)

# View report
print(missing_value_report(df))
```

### Different "Missing" Representations

Null is not the only way data can be missing:

```python
def detect_missing_representations(df: pl.DataFrame, col: str) -> dict:
    """Detect various representations of missing values."""
    if df.schema[col] != pl.Utf8:
        return {"null_count": df.select(pl.col(col).null_count()).item()}
    
    text = pl.col(col)
    
    return df.select(
        text.null_count().alias("null"),
        (text == "").sum().alias("empty_string"),
        text.str.to_lowercase().is_in(["n/a", "na", "null", "none"]).sum().alias("na_variants"),
        (text == "-").sum().alias("dash"),
        (text == "0").sum().alias("zero_string"),
        text.str.contains(r"^\s+$").sum().alias("whitespace_only"),
        text.str.to_lowercase().is_in(["unknown", "tbd", "pending"]).sum().alias("placeholder"),
    ).to_dicts()[0]
```

### Missingness Correlations

When two columns are missing together, it's often meaningful:

```python
def missing_correlation(df: pl.DataFrame, col_a: str, col_b: str) -> dict:
    """Check if missingness in two columns is correlated."""
    both_missing = df.filter(
        pl.col(col_a).is_null() & pl.col(col_b).is_null()
    ).height
    
    a_missing = df.select(pl.col(col_a).null_count()).item()
    b_missing = df.select(pl.col(col_b).null_count()).item()
    
    # If they're always missing together, correlation is high
    if a_missing > 0 and b_missing > 0:
        overlap_ratio = both_missing / min(a_missing, b_missing)
    else:
        overlap_ratio = 0
    
    return {
        "a_missing": a_missing,
        "b_missing": b_missing,
        "both_missing": both_missing,
        "overlap_ratio": overlap_ratio,
        "interpretation": "Possibly systematic" if overlap_ratio > 0.8 else "Likely independent"
    }
```

---

## 5. Multivariate Profiling

Multivariate profiling examines relationships between columns.

### Cross-Column Relationships

Check logical relationships:

```python
def check_date_order(df: pl.DataFrame, start_col: str, end_col: str) -> dict:
    """Check that start dates come before end dates."""
    violations = df.filter(pl.col(start_col) > pl.col(end_col))
    
    return {
        "total_rows": df.height,
        "violations": violations.height,
        "violation_pct": violations.height / df.height * 100,
        "sample_violations": violations.head(5).to_dicts(),
    }

def check_value_consistency(df: pl.DataFrame, col_a: str, col_b: str) -> dict:
    """Check if values in col_a consistently map to col_b."""
    # For each unique value in A, how many distinct values in B?
    mapping = (
        df.group_by(col_a)
        .agg(pl.col(col_b).n_unique().alias("distinct_b"))
        .filter(pl.col("distinct_b") > 1)
    )
    
    return {
        "inconsistent_mappings": mapping.height,
        "examples": mapping.head(10).to_dicts(),
    }
```

### Duplicate Detection

**Single-column duplicates:**

```python
def find_duplicates(df: pl.DataFrame, key_cols: list) -> dict:
    """Find duplicate rows based on key columns."""
    dup_counts = (
        df.group_by(key_cols)
        .agg(pl.count().alias("count"))
        .filter(pl.col("count") > 1)
    )
    
    total_duplicates = dup_counts.select(
        (pl.col("count") - 1).sum()
    ).item() or 0
    
    return {
        "duplicate_groups": dup_counts.height,
        "total_duplicate_rows": total_duplicates,
        "duplicate_pct": total_duplicates / df.height * 100,
        "worst_offenders": dup_counts.sort("count", descending=True).head(10).to_dicts(),
    }
```

**Composite key duplicates:**

```python
# Check if order_id + line_item should be unique
dup_report = find_duplicates(df, ["order_id", "line_item"])
print(f"Duplicate order lines: {dup_report['total_duplicate_rows']}")
```

### Referential Integrity Checks

When one column references another (like foreign keys):

```python
def check_referential_integrity(
    df: pl.DataFrame,
    fk_col: str,
    reference_df: pl.DataFrame,
    pk_col: str
) -> dict:
    """Check that all values in fk_col exist in reference_df's pk_col."""
    valid_values = set(reference_df[pk_col].to_list())
    
    orphans = df.filter(
        ~pl.col(fk_col).is_in(valid_values) & pl.col(fk_col).is_not_null()
    )
    
    return {
        "total_rows": df.height,
        "orphan_rows": orphans.height,
        "orphan_pct": orphans.height / df.height * 100,
        "orphan_values": orphans[fk_col].unique().to_list()[:20],
    }

# Example: check that all status values are valid
valid_statuses = pl.DataFrame({"status": ["pending", "shipped", "delivered", "cancelled"]})
integrity = check_referential_integrity(orders_df, "status", valid_statuses, "status")
```

---

## 6. Building a Profiling Report

Combine individual profiles into a comprehensive report.

### Reusable Profiling Functions

Create a unified profiler:

```python
def profile_column(df: pl.DataFrame, col: str) -> dict:
    """Profile a single column based on its type."""
    dtype = df.schema[col]
    
    base_profile = {
        "column": col,
        "dtype": str(dtype),
        "count": df.height,
        "null_count": df.select(pl.col(col).null_count()).item(),
        "null_pct": df.select(pl.col(col).null_count()).item() / df.height * 100,
        "unique_count": df.select(pl.col(col).n_unique()).item(),
    }
    
    if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
        base_profile.update(profile_numeric(df, col))
    elif dtype == pl.Utf8:
        base_profile.update(profile_text(df, col))
    elif dtype in [pl.Date, pl.Datetime]:
        base_profile.update(profile_date(df, col))
    
    return base_profile

def profile_dataframe(df: pl.DataFrame) -> list:
    """Profile all columns in a dataframe."""
    return [profile_column(df, col) for col in df.columns]
```

### Automated Report Generation

Generate a markdown or JSON report:

```python
import json
from datetime import datetime

def generate_profile_report(df: pl.DataFrame, name: str) -> str:
    """Generate a markdown profiling report."""
    profiles = profile_dataframe(df)
    missing_report = missing_value_report(df)
    
    report = f"""# Data Profile Report: {name}

Generated: {datetime.now().isoformat()}

## Summary

- **Rows:** {df.height:,}
- **Columns:** {len(df.columns)}
- **Memory:** {df.estimated_size() / 1024 / 1024:.2f} MB

## Schema

| Column | Type | Nulls | Unique |
|--------|------|-------|--------|
"""
    
    for p in profiles:
        report += f"| {p['column']} | {p['dtype']} | {p['null_pct']:.1f}% | {p['unique_count']:,} |\n"
    
    report += "\n## Column Details\n\n"
    
    for p in profiles:
        report += f"### {p['column']}\n\n"
        report += f"- Type: {p['dtype']}\n"
        report += f"- Nulls: {p['null_count']:,} ({p['null_pct']:.1f}%)\n"
        report += f"- Unique: {p['unique_count']:,}\n"
        
        if "mean" in p:
            report += f"- Range: {p.get('min')} to {p.get('max')}\n"
            report += f"- Mean: {p.get('mean'):.2f}\n"
        
        report += "\n"
    
    return report

# Save report
report = generate_profile_report(df, "Orders Dataset")
with open("profile_report.md", "w") as f:
    f.write(report)
```

### What to Flag for Investigation

Prioritize issues that need attention:

```python
def generate_alerts(df: pl.DataFrame) -> list:
    """Generate prioritized alerts from profiling."""
    alerts = []
    
    for col in df.columns:
        dtype = df.schema[col]
        null_pct = df.select(pl.col(col).null_count()).item() / df.height * 100
        unique_count = df.select(pl.col(col).n_unique()).item()
        
        # High null percentage
        if null_pct > 50:
            alerts.append({
                "severity": "HIGH",
                "column": col,
                "issue": f"High null percentage: {null_pct:.1f}%",
            })
        elif null_pct > 10:
            alerts.append({
                "severity": "MEDIUM",
                "column": col,
                "issue": f"Notable null percentage: {null_pct:.1f}%",
            })
        
        # Suspiciously low uniqueness (possible constant)
        if unique_count == 1:
            alerts.append({
                "severity": "MEDIUM",
                "column": col,
                "issue": "Column has only one unique value (constant)",
            })
        
        # Type mismatch indicators
        if dtype == pl.Utf8:
            numeric_check = detect_numeric_strings(df.select(col))
            if col in numeric_check:
                alerts.append({
                    "severity": "HIGH",
                    "column": col,
                    "issue": "Numeric data stored as string",
                })
    
    return sorted(alerts, key=lambda x: {"HIGH": 0, "MEDIUM": 1, "LOW": 2}[x["severity"]])
```

---

## Summary

Data profiling is the foundation of reliable data engineering.

**Schema profiling** reveals type mismatches and structural issues before they cause failures.

**Univariate profiling** examines each column's distribution, range, and quality indicators.

**Missing value analysis** distinguishes random gaps from systematic data collection problems.

**Multivariate profiling** checks relationships, duplicates, and referential integrity.

**Automated reports** make profiling a standard, repeatable part of every pipeline.

Profile first, transform second. The cost of profiling is minutes; the cost of skipping it is hours of debugging.

---

## Interview Framing

**"How do you approach a new dataset?"**

"First, I profile before any transformation. Schema check to verify types match expectations — numeric data stored as strings is the most common issue. Univariate profiling for distributions: min, max, percentiles for numbers; length and pattern analysis for text. Missing value analysis to distinguish random nulls from systematic gaps. Then multivariate checks: duplicates, referential integrity, logical constraints like dates in order. I generate an automated report that flags issues by severity. Only after profiling do I start cleaning."

**"How do you detect data quality issues?"**

"Profiling catches most issues upfront. Numeric strings reveal themselves when type casting fails. Outliers show up in the gap between p99 and max. Systematic missingness appears when you correlate null patterns across columns. Duplicates surface in composite key analysis. I also check for 'soft' problems: placeholder values like 'N/A' or 'TBD' that aren't null but aren't real data either. The key is systematic examination, not spot-checking."
