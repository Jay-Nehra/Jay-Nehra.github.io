# Quality Validation

Quality validation is the systematic verification that data meets defined standards. It answers the question: "Is this data good enough for its intended use?"

Validation is not cleaning. Cleaning transforms data. Validation checks whether data — cleaned or not — meets requirements. Validation produces verdicts: pass, fail, or warning. Those verdicts drive decisions about what to use, what to fix, and what to reject.

This guide covers validation for business data using polars, with practical examples from financial transactions and order processing.

---

## 1. Quality Dimensions

Data quality is not one thing. It has dimensions that must be measured separately.

### The Five Dimensions

**Completeness:** Is the data present? Are required fields populated? Are there gaps?

**Accuracy:** Is the data correct? Does it reflect reality? Are values within expected bounds?

**Consistency:** Is the data uniform? Do related values agree? Are formats standard?

**Timeliness:** Is the data current? Is it from the right time period? Is it stale?

**Uniqueness:** Is each record distinct? Are there unwanted duplicates? Are identifiers unique?

Each dimension matters differently depending on use case:

| Use Case | Most Critical Dimensions |
|----------|-------------------------|
| Financial reporting | Accuracy, Completeness |
| Customer analytics | Uniqueness, Consistency |
| Real-time dashboards | Timeliness, Completeness |
| Machine learning training | Accuracy, Uniqueness |

### Defining "Good Enough"

Perfect data doesn't exist. The question is: what quality level is acceptable?

Define thresholds:
- **Completeness:** At least 95% of orders have shipping addresses
- **Accuracy:** No negative amounts (0 tolerance)
- **Consistency:** Date format violations below 1%
- **Uniqueness:** Zero duplicate order IDs

These thresholds are business decisions, not technical ones. A data engineer can measure quality; stakeholders define acceptable levels.

```python
# Define quality thresholds
QUALITY_THRESHOLDS = {
    "order_amount": {
        "null_rate_max": 0.01,      # Max 1% nulls
        "negative_count_max": 0,     # No negatives
    },
    "order_date": {
        "null_rate_max": 0.0,        # No nulls allowed
        "future_count_max": 0,       # No future dates
    },
    "order_id": {
        "duplicate_rate_max": 0.0,   # Must be unique
    },
}
```

### Quality as a Spectrum

Quality is not binary. A dataset can be:
- **High quality:** Passes all checks, ready for production
- **Acceptable:** Minor issues, usable with caveats
- **Degraded:** Significant issues, needs review before use
- **Failed:** Critical issues, cannot be used

Map validation results to these levels:

```python
def classify_quality(validation_results: dict) -> str:
    """Classify overall quality level based on validation results."""
    critical_failures = sum(1 for r in validation_results.values() if r["severity"] == "CRITICAL" and not r["passed"])
    warnings = sum(1 for r in validation_results.values() if r["severity"] == "WARNING" and not r["passed"])
    
    if critical_failures > 0:
        return "FAILED"
    elif warnings > 3:
        return "DEGRADED"
    elif warnings > 0:
        return "ACCEPTABLE"
    else:
        return "HIGH_QUALITY"
```

---

## 2. Rule Types

Validation rules check specific conditions. Different rule types catch different problems.

### Type Rules

Verify columns have the expected data type:

```python
def validate_type(df: pl.DataFrame, col: str, expected_type: pl.DataType) -> dict:
    """Validate column has expected type."""
    actual_type = df.schema[col]
    passed = actual_type == expected_type
    
    return {
        "rule": "type_check",
        "column": col,
        "passed": passed,
        "expected": str(expected_type),
        "actual": str(actual_type),
        "severity": "CRITICAL",
    }
```

### Range Rules

Verify numeric values fall within expected bounds:

```python
def validate_range(
    df: pl.DataFrame,
    col: str,
    min_val: float = None,
    max_val: float = None
) -> dict:
    """Validate numeric column is within range."""
    violations = df.filter(
        (pl.col(col) < min_val if min_val is not None else pl.lit(False)) |
        (pl.col(col) > max_val if max_val is not None else pl.lit(False))
    )
    
    return {
        "rule": "range_check",
        "column": col,
        "passed": violations.height == 0,
        "violation_count": violations.height,
        "violation_rate": violations.height / df.height,
        "min_expected": min_val,
        "max_expected": max_val,
        "severity": "CRITICAL",
    }

# Example: amounts must be positive and under $1M
result = validate_range(df, "order_amount", min_val=0, max_val=1_000_000)
```

### Pattern Rules

Verify text matches expected patterns:

```python
def validate_pattern(
    df: pl.DataFrame,
    col: str,
    pattern: str,
    description: str = "pattern"
) -> dict:
    """Validate text column matches regex pattern."""
    non_null = df.filter(pl.col(col).is_not_null())
    matches = non_null.filter(pl.col(col).str.contains(pattern))
    violations = non_null.height - matches.height
    
    return {
        "rule": "pattern_check",
        "column": col,
        "pattern": pattern,
        "description": description,
        "passed": violations == 0,
        "violation_count": violations,
        "violation_rate": violations / non_null.height if non_null.height > 0 else 0,
        "severity": "WARNING",
    }

# Example: email format
email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
result = validate_pattern(df, "email", email_pattern, "valid email format")

# Example: product code format (ABC-12345)
sku_pattern = r"^[A-Z]{3}-\d{5}$"
result = validate_pattern(df, "product_code", sku_pattern, "SKU format")
```

### Referential Rules

Verify values exist in a reference set:

```python
def validate_referential(
    df: pl.DataFrame,
    col: str,
    valid_values: list,
    description: str = "allowed values"
) -> dict:
    """Validate column values are in allowed set."""
    non_null = df.filter(pl.col(col).is_not_null())
    invalid = non_null.filter(~pl.col(col).is_in(valid_values))
    
    return {
        "rule": "referential_check",
        "column": col,
        "description": description,
        "passed": invalid.height == 0,
        "violation_count": invalid.height,
        "invalid_values": invalid[col].unique().to_list()[:10],
        "severity": "CRITICAL",
    }

# Example: status must be from allowed list
valid_statuses = ["pending", "processing", "shipped", "delivered", "cancelled"]
result = validate_referential(df, "status", valid_statuses, "valid order status")
```

### Cross-Column Rules

Verify relationships between columns:

```python
def validate_cross_column(
    df: pl.DataFrame,
    condition: pl.Expr,
    description: str
) -> dict:
    """Validate a condition across multiple columns."""
    violations = df.filter(~condition)
    
    return {
        "rule": "cross_column_check",
        "description": description,
        "passed": violations.height == 0,
        "violation_count": violations.height,
        "violation_rate": violations.height / df.height,
        "severity": "CRITICAL",
    }

# Example: end_date >= start_date
result = validate_cross_column(
    df,
    pl.col("end_date") >= pl.col("start_date"),
    "end_date must be after start_date"
)

# Example: if status is 'shipped', ship_date must exist
result = validate_cross_column(
    df,
    ~((pl.col("status") == "shipped") & pl.col("ship_date").is_null()),
    "shipped orders must have ship_date"
)

# Example: discount cannot exceed total
result = validate_cross_column(
    df,
    pl.col("discount_amount") <= pl.col("order_amount"),
    "discount cannot exceed order amount"
)
```

---

## 3. Expressing Rules in Polars

Polars expressions make validation declarative and efficient.

### Boolean Expressions as Validators

Every validation rule is a boolean expression:

```python
# Each row either passes (True) or fails (False)
is_valid_amount = pl.col("amount") > 0
is_valid_email = pl.col("email").str.contains(r"^.+@.+\..+$")
is_valid_date = pl.col("order_date") <= pl.lit(date.today())

# Combine for row-level validation
is_valid_row = is_valid_amount & is_valid_email & is_valid_date

# Apply to dataframe
df = df.with_columns(
    is_valid_row.alias("is_valid")
)
```

### Composing Complex Rules

Build rules from components:

```python
from dataclasses import dataclass
from typing import Callable

@dataclass
class ValidationRule:
    name: str
    expression: pl.Expr
    severity: str  # CRITICAL, WARNING, INFO
    description: str

def create_validation_suite(rules: list[ValidationRule]) -> Callable:
    """Create a validation function from rules."""
    
    def validate(df: pl.DataFrame) -> dict:
        results = {}
        
        for rule in rules:
            # Count violations
            violations = df.filter(~rule.expression)
            
            results[rule.name] = {
                "description": rule.description,
                "severity": rule.severity,
                "passed": violations.height == 0,
                "violation_count": violations.height,
                "violation_rate": violations.height / df.height,
            }
        
        return results
    
    return validate

# Define rules
order_rules = [
    ValidationRule(
        name="positive_amount",
        expression=pl.col("amount") > 0,
        severity="CRITICAL",
        description="Order amount must be positive"
    ),
    ValidationRule(
        name="valid_status",
        expression=pl.col("status").is_in(["pending", "shipped", "delivered"]),
        severity="CRITICAL",
        description="Status must be valid"
    ),
    ValidationRule(
        name="reasonable_amount",
        expression=pl.col("amount") < 100000,
        severity="WARNING",
        description="Amount should be under $100k"
    ),
]

# Create and run validator
validate_orders = create_validation_suite(order_rules)
results = validate_orders(df)
```

### Rule Libraries and Reusability

Build a library of common rules:

```python
class CommonValidators:
    """Library of reusable validation expressions."""
    
    @staticmethod
    def not_null(col: str) -> pl.Expr:
        return pl.col(col).is_not_null()
    
    @staticmethod
    def positive(col: str) -> pl.Expr:
        return pl.col(col) > 0
    
    @staticmethod
    def non_negative(col: str) -> pl.Expr:
        return pl.col(col) >= 0
    
    @staticmethod
    def in_range(col: str, min_val: float, max_val: float) -> pl.Expr:
        return (pl.col(col) >= min_val) & (pl.col(col) <= max_val)
    
    @staticmethod
    def not_empty(col: str) -> pl.Expr:
        return (pl.col(col).is_not_null()) & (pl.col(col) != "")
    
    @staticmethod
    def matches_pattern(col: str, pattern: str) -> pl.Expr:
        return pl.col(col).str.contains(pattern)
    
    @staticmethod
    def in_set(col: str, valid_values: list) -> pl.Expr:
        return pl.col(col).is_in(valid_values)
    
    @staticmethod
    def date_not_future(col: str) -> pl.Expr:
        return pl.col(col) <= pl.lit(date.today())
    
    @staticmethod
    def date_in_range(col: str, min_date: date, max_date: date) -> pl.Expr:
        return (pl.col(col) >= min_date) & (pl.col(col) <= max_date)

# Usage
v = CommonValidators
rules = [
    ValidationRule("amount_positive", v.positive("amount"), "CRITICAL", "Amount > 0"),
    ValidationRule("email_present", v.not_empty("email"), "WARNING", "Email required"),
    ValidationRule("date_valid", v.date_not_future("order_date"), "CRITICAL", "No future dates"),
]
```

---

## 4. Validation Execution

How you run validation affects usability and performance.

### Row-Level vs Dataset-Level Validation

**Row-level:** Each row passes or fails independently.

```python
def row_level_validation(df: pl.DataFrame, rules: list) -> pl.DataFrame:
    """Add validation columns for each row."""
    for rule in rules:
        df = df.with_columns(
            rule.expression.alias(f"valid_{rule.name}")
        )
    
    # Add overall validity
    valid_cols = [f"valid_{rule.name}" for rule in rules]
    df = df.with_columns(
        pl.all_horizontal(*[pl.col(c) for c in valid_cols]).alias("is_valid")
    )
    
    return df
```

**Dataset-level:** Aggregate checks on the entire dataset.

```python
def dataset_level_validation(df: pl.DataFrame) -> dict:
    """Validate dataset-level properties."""
    return {
        "row_count": {
            "value": df.height,
            "passed": df.height > 0,
            "severity": "CRITICAL",
        },
        "null_rate_overall": {
            "value": df.null_count().sum_horizontal().sum() / (df.height * len(df.columns)),
            "passed": df.null_count().sum_horizontal().sum() / (df.height * len(df.columns)) < 0.1,
            "severity": "WARNING",
        },
        "duplicate_rate": {
            "value": 1 - df.unique().height / df.height,
            "passed": df.unique().height == df.height,
            "severity": "WARNING",
        },
    }
```

### Fail-Fast vs Collect-All-Errors

**Fail-fast:** Stop on first failure (good for pipelines where one failure blocks everything).

```python
def validate_fail_fast(df: pl.DataFrame, rules: list) -> tuple[bool, dict]:
    """Validate and stop on first critical failure."""
    for rule in rules:
        violations = df.filter(~rule.expression)
        if violations.height > 0 and rule.severity == "CRITICAL":
            return False, {
                "failed_rule": rule.name,
                "violation_count": violations.height,
                "sample_violations": violations.head(5).to_dicts(),
            }
    return True, {}
```

**Collect-all-errors:** Run all validations, report everything (good for diagnostics).

```python
def validate_collect_all(df: pl.DataFrame, rules: list) -> dict:
    """Validate all rules and collect all results."""
    results = {}
    
    for rule in rules:
        violations = df.filter(~rule.expression)
        results[rule.name] = {
            "passed": violations.height == 0,
            "violation_count": violations.height,
            "severity": rule.severity,
        }
    
    results["summary"] = {
        "total_rules": len(rules),
        "passed": sum(1 for r in results.values() if isinstance(r, dict) and r.get("passed", False)),
        "failed": sum(1 for r in results.values() if isinstance(r, dict) and not r.get("passed", True)),
    }
    
    return results
```

### Validation Reports

Generate human-readable reports:

```python
def generate_validation_report(results: dict, dataset_name: str) -> str:
    """Generate markdown validation report."""
    report = f"# Validation Report: {dataset_name}\n\n"
    report += f"Generated: {datetime.now().isoformat()}\n\n"
    
    # Summary
    passed = sum(1 for r in results.values() if isinstance(r, dict) and r.get("passed"))
    failed = sum(1 for r in results.values() if isinstance(r, dict) and not r.get("passed", True))
    
    report += f"## Summary\n\n"
    report += f"- **Passed:** {passed}\n"
    report += f"- **Failed:** {failed}\n\n"
    
    # Details
    report += "## Rule Results\n\n"
    report += "| Rule | Status | Violations | Severity |\n"
    report += "|------|--------|------------|----------|\n"
    
    for name, result in results.items():
        if isinstance(result, dict) and "passed" in result:
            status = "✓ Pass" if result["passed"] else "✗ Fail"
            violations = result.get("violation_count", 0)
            severity = result.get("severity", "INFO")
            report += f"| {name} | {status} | {violations} | {severity} |\n"
    
    return report
```

---

## 5. Severity Levels

Not all validation failures are equal.

### Critical (Blocks Processing)

Failures that make data unusable:
- Primary key duplicates
- Required fields null
- Values outside hard constraints (negative amounts)
- Type coercion failures

```python
critical_rules = [
    ValidationRule("order_id_unique", 
                   pl.col("order_id").is_unique(), 
                   "CRITICAL",
                   "Order ID must be unique"),
    ValidationRule("amount_positive",
                   pl.col("amount") > 0,
                   "CRITICAL",
                   "Amount must be positive"),
]
```

### Warning (Flag but Proceed)

Issues that need attention but don't block:
- Unusually high or low values
- Missing optional fields
- Minor format inconsistencies

```python
warning_rules = [
    ValidationRule("amount_reasonable",
                   pl.col("amount") < 100000,
                   "WARNING",
                   "Unusually high amount"),
    ValidationRule("phone_present",
                   pl.col("phone").is_not_null(),
                   "WARNING",
                   "Phone number recommended"),
]
```

### Info (Log for Awareness)

Observations that aren't problems:
- Statistics outside typical ranges
- New category values appearing
- Distribution shifts

```python
info_checks = {
    "new_categories": df.filter(~pl.col("status").is_in(known_statuses)),
    "high_null_columns": [c for c in df.columns if df[c].null_count() / df.height > 0.5],
}
```

### Implementing Severity-Based Decisions

```python
def make_processing_decision(validation_results: dict) -> str:
    """Decide whether to proceed based on validation results."""
    critical_failures = [
        name for name, result in validation_results.items()
        if isinstance(result, dict) 
        and result.get("severity") == "CRITICAL" 
        and not result.get("passed", True)
    ]
    
    if critical_failures:
        return f"ABORT: Critical failures in {critical_failures}"
    
    warnings = [
        name for name, result in validation_results.items()
        if isinstance(result, dict)
        and result.get("severity") == "WARNING"
        and not result.get("passed", True)
    ]
    
    if warnings:
        return f"PROCEED_WITH_CAUTION: Warnings in {warnings}"
    
    return "PROCEED: All validations passed"
```

---

## 6. Statistical Validation

Beyond rule-based checks, statistical validation catches drift and anomalies.

### Distribution Drift Detection

Compare current data to a baseline:

```python
def detect_distribution_drift(
    current_df: pl.DataFrame,
    baseline_stats: dict,
    col: str,
    threshold: float = 0.2
) -> dict:
    """Detect if distribution has shifted significantly."""
    current_stats = current_df.select(
        pl.col(col).mean().alias("mean"),
        pl.col(col).std().alias("std"),
        pl.col(col).quantile(0.5).alias("median"),
    ).to_dicts()[0]
    
    # Compare means (relative change)
    mean_drift = abs(current_stats["mean"] - baseline_stats["mean"]) / baseline_stats["mean"]
    std_drift = abs(current_stats["std"] - baseline_stats["std"]) / baseline_stats["std"]
    
    return {
        "column": col,
        "mean_drift": mean_drift,
        "std_drift": std_drift,
        "drift_detected": mean_drift > threshold or std_drift > threshold,
        "current": current_stats,
        "baseline": baseline_stats,
    }
```

### Anomaly Detection in Aggregates

Check for unusual patterns in aggregated data:

```python
def detect_aggregate_anomalies(
    df: pl.DataFrame,
    group_col: str,
    value_col: str,
    z_threshold: float = 3.0
) -> pl.DataFrame:
    """Find groups with anomalous aggregates."""
    aggregates = df.group_by(group_col).agg(
        pl.col(value_col).sum().alias("total"),
        pl.col(value_col).count().alias("count"),
    )
    
    # Calculate z-scores
    mean_total = aggregates["total"].mean()
    std_total = aggregates["total"].std()
    
    aggregates = aggregates.with_columns(
        ((pl.col("total") - mean_total) / std_total).abs().alias("z_score")
    )
    
    # Flag anomalies
    anomalies = aggregates.filter(pl.col("z_score") > z_threshold)
    
    return anomalies
```

### Baseline Comparison

Store and compare against known-good baselines:

```python
import json

def create_baseline(df: pl.DataFrame, numeric_cols: list) -> dict:
    """Create a statistical baseline from known-good data."""
    baseline = {}
    
    for col in numeric_cols:
        stats = df.select(
            pl.col(col).count().alias("count"),
            pl.col(col).null_count().alias("null_count"),
            pl.col(col).mean().alias("mean"),
            pl.col(col).std().alias("std"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
            pl.col(col).quantile(0.25).alias("p25"),
            pl.col(col).quantile(0.75).alias("p75"),
        ).to_dicts()[0]
        baseline[col] = stats
    
    return baseline

def compare_to_baseline(df: pl.DataFrame, baseline: dict) -> dict:
    """Compare current data to baseline."""
    issues = []
    
    for col, base_stats in baseline.items():
        current = df.select(
            pl.col(col).mean().alias("mean"),
            pl.col(col).min().alias("min"),
            pl.col(col).max().alias("max"),
        ).to_dicts()[0]
        
        # Check for significant changes
        if current["mean"] < base_stats["mean"] * 0.8 or current["mean"] > base_stats["mean"] * 1.2:
            issues.append(f"{col}: mean shifted from {base_stats['mean']:.2f} to {current['mean']:.2f}")
        
        if current["min"] < base_stats["min"] * 0.5:
            issues.append(f"{col}: min dropped from {base_stats['min']} to {current['min']}")
        
        if current["max"] > base_stats["max"] * 2:
            issues.append(f"{col}: max increased from {base_stats['max']} to {current['max']}")
    
    return {"issues": issues, "passed": len(issues) == 0}
```

---

## 7. Validation in Pipelines

Validation should be integrated into data pipelines at multiple points.

### Pre-Load Validation (Schema Checks)

Check schema before loading full data:

```python
def validate_schema_before_load(filepath: str, expected_schema: dict) -> dict:
    """Validate file schema without loading full data."""
    # Read just the header
    sample = pl.read_csv(filepath, n_rows=0)
    
    issues = []
    
    # Check columns exist
    missing = set(expected_schema.keys()) - set(sample.columns)
    if missing:
        issues.append(f"Missing columns: {missing}")
    
    extra = set(sample.columns) - set(expected_schema.keys())
    if extra:
        issues.append(f"Unexpected columns: {extra}")
    
    return {
        "passed": len(issues) == 0,
        "issues": issues,
        "actual_columns": sample.columns,
    }
```

### Post-Load Validation (Content Checks)

Full validation after loading:

```python
def validate_loaded_data(df: pl.DataFrame, rules: list) -> dict:
    """Run full validation on loaded data."""
    results = validate_collect_all(df, rules)
    results["row_count"] = df.height
    results["null_summary"] = {
        col: df[col].null_count() / df.height 
        for col in df.columns
    }
    return results
```

### Continuous Validation (Monitoring)

For ongoing pipelines, validate each batch:

```python
from datetime import datetime

class PipelineValidator:
    """Validator for ongoing data pipelines."""
    
    def __init__(self, rules: list, baseline: dict = None):
        self.rules = rules
        self.baseline = baseline
        self.history = []
    
    def validate_batch(self, df: pl.DataFrame, batch_id: str) -> dict:
        """Validate a batch and track results."""
        results = validate_collect_all(df, self.rules)
        results["batch_id"] = batch_id
        results["timestamp"] = datetime.now().isoformat()
        results["row_count"] = df.height
        
        # Compare to baseline if available
        if self.baseline:
            drift = compare_to_baseline(df, self.baseline)
            results["drift_check"] = drift
        
        self.history.append(results)
        return results
    
    def get_quality_trend(self) -> pl.DataFrame:
        """Get quality metrics over time."""
        records = []
        for h in self.history:
            passed = sum(1 for r in h.values() if isinstance(r, dict) and r.get("passed"))
            total = sum(1 for r in h.values() if isinstance(r, dict) and "passed" in r)
            records.append({
                "batch_id": h["batch_id"],
                "timestamp": h["timestamp"],
                "pass_rate": passed / total if total > 0 else 0,
                "row_count": h["row_count"],
            })
        return pl.DataFrame(records)
```

### Integration Pattern

```python
def data_pipeline_with_validation(
    source_path: str,
    rules: list,
    output_path: str
) -> dict:
    """Example pipeline with validation gates."""
    
    # Gate 1: Schema check
    schema_result = validate_schema_before_load(source_path, EXPECTED_SCHEMA)
    if not schema_result["passed"]:
        return {"status": "FAILED", "stage": "schema_check", "details": schema_result}
    
    # Load
    df = pl.read_csv(source_path)
    
    # Gate 2: Content validation
    validation_result = validate_collect_all(df, rules)
    
    # Check critical failures
    critical_failures = [
        name for name, result in validation_result.items()
        if isinstance(result, dict) and result.get("severity") == "CRITICAL" and not result.get("passed", True)
    ]
    
    if critical_failures:
        return {
            "status": "FAILED",
            "stage": "content_validation",
            "critical_failures": critical_failures,
            "details": validation_result,
        }
    
    # Gate 3: Statistical checks (warning only)
    stat_result = compare_to_baseline(df, BASELINE)
    
    # Proceed with warnings logged
    df.write_parquet(output_path)
    
    return {
        "status": "SUCCESS",
        "validation": validation_result,
        "statistical_check": stat_result,
        "rows_written": df.height,
    }
```

---

## Summary

Quality validation is systematic verification against defined standards.

**Quality dimensions** — completeness, accuracy, consistency, timeliness, uniqueness — are measured separately.

**Rule types** — type, range, pattern, referential, cross-column — catch different problems.

**Severity levels** determine what blocks processing versus what is logged.

**Statistical validation** catches drift and anomalies that rules miss.

**Pipeline integration** places validation gates at schema, content, and statistical levels.

---

## Interview Framing

**"How do you ensure data quality in a pipeline?"**

"I implement validation at multiple gates. Pre-load schema checks verify the file structure before reading. Post-load content validation runs rule-based checks: type constraints, range bounds, referential integrity, and cross-column logic. I also run statistical validation comparing to baselines to catch distribution drift. Rules have severity levels — critical failures block the pipeline, warnings are logged and monitored. Results are tracked over time to catch gradual degradation."

**"What's the difference between data cleaning and data validation?"**

"Cleaning transforms data — fixing formats, normalizing values, handling missing data. Validation checks whether data meets requirements — it doesn't change anything, it produces verdicts. You clean first, then validate to confirm the cleaning worked. In a pipeline, validation gates decide whether to proceed or abort. Cleaning is about making data usable; validation is about proving it's acceptable."
