# Correction Strategies

When validation finds problems, you must decide what to do. Correction is not simply "fix everything" — it's a systematic process of deciding what can be fixed, what should be fixed, and how to handle what cannot be fixed.

This guide covers correction strategies for business data: when to auto-correct, when to impute, when to quarantine, and when to reject. The goal is making these decisions consistently and transparently.

---

## 1. The Correction Decision Framework

Not every data problem should be corrected automatically. Before fixing anything, ask three questions.

### Can It Be Fixed Automatically?

Some problems have deterministic fixes:
- "$1,234.56" → 1234.56 (format stripping)
- "YES" → "Y" (value mapping)
- Leading/trailing whitespace (trimming)

Others require context or judgment:
- Which "John Smith" is this? (entity resolution)
- Is this outlier a data error or a real extreme? (domain knowledge)
- What date did they mean by "next Tuesday"? (ambiguity)

If a fix requires human judgment, it cannot be automated reliably.

### Should It Be Fixed Automatically?

Even when you can fix something, you might not want to:

**Risk of wrong correction:** Changing "1000" to "100" based on an outlier rule might destroy a valid large order.

**Audit requirements:** Some industries require original data preservation. Automated changes may violate compliance.

**Downstream impact:** Correcting data that feeds multiple systems can propagate changes unexpectedly.

**Masking problems:** Auto-correcting silently hides issues that should be escalated to data producers.

The question is not "can I fix this?" but "what are the consequences of fixing this wrong?"

### Risk of Over-Correction

Over-correction happens when you:
- Apply fixes that are sometimes wrong
- Fix things that weren't actually broken
- Make data "too clean" (lose legitimate variation)
- Chain corrections that compound errors

```python
# Example: over-aggressive outlier correction
# This "fixes" legitimate high-value orders
df = df.with_columns(
    pl.when(pl.col("amount") > 10000)
    .then(pl.lit(10000))  # Cap at 10k — but what about enterprise deals?
    .otherwise(pl.col("amount"))
)
```

The principle: be conservative. When in doubt, flag rather than fix.

---

## 2. Correction Categories

Classify corrections by how reliably they can be applied.

### Deterministic Corrections

Always fixable the same way. Zero ambiguity.

**Examples:**
- Whitespace normalization
- Case standardization
- Known format conversions (currency symbols, date formats)
- Lookup-based replacements (known typo → correct value)

```python
# Deterministic: these are always safe
def deterministic_corrections(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        # Whitespace — always trim
        pl.col("customer_name").str.strip_chars(),
        
        # Case — standard format
        pl.col("email").str.to_lowercase(),
        
        # Known mapping — typos we've validated
        pl.col("status").replace({
            "shiped": "shipped",
            "deliverd": "delivered",
            "cancled": "cancelled",
        }),
    )
```

Deterministic corrections can be applied automatically with high confidence.

### Probabilistic Corrections

Likely correct based on patterns, but not certain.

**Examples:**
- Fuzzy matching typos to known values
- Inferring missing values from related data
- Outlier adjustment based on statistical models

```python
# Probabilistic: usually right, sometimes wrong
from difflib import get_close_matches

def probabilistic_status_fix(value: str, valid_statuses: list, threshold: float = 0.85) -> str:
    """Fix status if it closely matches a known value."""
    if value in valid_statuses:
        return value
    
    matches = get_close_matches(value.lower(), [s.lower() for s in valid_statuses], n=1, cutoff=threshold)
    if matches:
        # Return original case version
        idx = [s.lower() for s in valid_statuses].index(matches[0])
        return valid_statuses[idx]
    
    return value  # No confident match, keep original
```

Probabilistic corrections should:
- Log what was changed
- Be reviewable
- Have confidence thresholds
- Have fallback behavior when confidence is low

### Manual Corrections

Require human judgment. Cannot be automated.

**Examples:**
- Entity resolution (which customer is this?)
- Business rule exceptions (is this outlier legitimate?)
- Ambiguous dates (01/02/03 — which format?)
- Missing context (what did the user mean?)

For manual corrections, the data pipeline should:
- Identify and extract records needing review
- Route to appropriate reviewers
- Provide context for decision-making
- Accept corrections back into the pipeline

```python
def extract_for_manual_review(df: pl.DataFrame, rules: list) -> tuple:
    """Separate data needing manual review."""
    needs_review = df.filter(
        # Complex condition for manual review
        (pl.col("amount") > 50000) |  # High-value — verify
        (pl.col("customer_id").is_null()) |  # Missing customer — research
        (pl.col("status") == "UNKNOWN")  # Unrecognized status
    )
    
    auto_processable = df.filter(
        ~(
            (pl.col("amount") > 50000) |
            (pl.col("customer_id").is_null()) |
            (pl.col("status") == "UNKNOWN")
        )
    )
    
    return auto_processable, needs_review
```

---

## 3. Automated Correction Patterns

When correction is appropriate, use consistent patterns.

### Lookup-Based Corrections

Map known incorrect values to correct values.

```python
def apply_lookup_corrections(
    df: pl.DataFrame,
    col: str,
    corrections: dict
) -> pl.DataFrame:
    """Apply corrections from a lookup table."""
    # Case-insensitive matching
    lower_corrections = {k.lower(): v for k, v in corrections.items()}
    
    return df.with_columns(
        pl.col(col)
        .str.to_lowercase()
        .replace(lower_corrections)
        .alias(f"{col}_corrected"),
        
        # Track what was corrected
        pl.col(col).str.to_lowercase().is_in(list(lower_corrections.keys())).alias(f"{col}_was_corrected")
    )

# Usage: known typos in category values
category_corrections = {
    "elctronics": "Electronics",
    "electroncs": "Electronics",
    "housewear": "Housewares",
    "housewears": "Housewares",
}

df = apply_lookup_corrections(df, "category", category_corrections)
```

### Rule-Based Corrections

Apply transformations based on patterns.

```python
def apply_rule_based_corrections(df: pl.DataFrame) -> pl.DataFrame:
    """Apply systematic rule-based corrections."""
    return df.with_columns(
        # Negative amounts in parentheses
        pl.when(pl.col("amount_str").str.contains(r"^\(.*\)$"))
        .then(
            pl.lit("-") + pl.col("amount_str").str.replace_all(r"[()]", "")
        )
        .otherwise(pl.col("amount_str"))
        .alias("amount_str_fixed"),
        
        # Phone numbers: strip to digits only
        pl.col("phone")
        .str.replace_all(r"[^\d]", "")
        .alias("phone_normalized"),
        
        # Dates: standardize separators
        pl.col("date_str")
        .str.replace_all(r"[/\-.]", "-")
        .alias("date_str_normalized"),
    )
```

### Inference-Based Corrections

Fill missing values from related data.

```python
def infer_missing_from_related(df: pl.DataFrame) -> pl.DataFrame:
    """Infer missing values from related columns or rows."""
    
    # Example 1: Infer country from phone prefix
    df = df.with_columns(
        pl.when(pl.col("country").is_null() & pl.col("phone").str.starts_with("+1"))
        .then(pl.lit("US"))
        .when(pl.col("country").is_null() & pl.col("phone").str.starts_with("+44"))
        .then(pl.lit("UK"))
        .otherwise(pl.col("country"))
        .alias("country_inferred")
    )
    
    # Example 2: Infer shipping date from status
    df = df.with_columns(
        pl.when(
            pl.col("ship_date").is_null() & 
            pl.col("status").is_in(["shipped", "delivered"])
        )
        .then(pl.col("order_date"))  # Best guess: shipped same day
        .otherwise(pl.col("ship_date"))
        .alias("ship_date_inferred")
    )
    
    return df
```

---

## 4. Imputation Strategies

Imputation fills missing values. Different strategies suit different data types.

### Numeric Imputation

**Mean:** Replace nulls with column mean.
- Good for: normally distributed data with random missingness
- Bad for: skewed distributions, systematic missingness

```python
def impute_mean(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mean_val = df.select(pl.col(col).mean()).item()
    return df.with_columns(
        pl.col(col).fill_null(mean_val).alias(f"{col}_imputed")
    )
```

**Median:** Replace nulls with column median.
- Good for: skewed distributions, outlier-resistant
- Bad for: multimodal distributions

```python
def impute_median(df: pl.DataFrame, col: str) -> pl.DataFrame:
    median_val = df.select(pl.col(col).median()).item()
    return df.with_columns(
        pl.col(col).fill_null(median_val).alias(f"{col}_imputed")
    )
```

**Forward/Backward Fill:** Use previous or next value.
- Good for: time series, sequential data
- Bad for: random ordering, independent observations

```python
def impute_forward_fill(df: pl.DataFrame, col: str, order_col: str) -> pl.DataFrame:
    return df.sort(order_col).with_columns(
        pl.col(col).forward_fill().alias(f"{col}_imputed")
    )
```

**Group-Based:** Use group statistics.
- Good for: when missingness relates to a category
- Bad for: small groups with no values

```python
def impute_group_mean(df: pl.DataFrame, value_col: str, group_col: str) -> pl.DataFrame:
    group_means = df.group_by(group_col).agg(
        pl.col(value_col).mean().alias("group_mean")
    )
    
    return df.join(group_means, on=group_col).with_columns(
        pl.coalesce(pl.col(value_col), pl.col("group_mean")).alias(f"{value_col}_imputed")
    ).drop("group_mean")
```

### Categorical Imputation

**Mode:** Replace with most common value.

```python
def impute_mode(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mode_val = (
        df.group_by(col)
        .count()
        .filter(pl.col(col).is_not_null())
        .sort("count", descending=True)
        .head(1)[col]
        .item()
    )
    return df.with_columns(
        pl.col(col).fill_null(mode_val).alias(f"{col}_imputed")
    )
```

**"Unknown" Category:** Explicit missing indicator.

```python
def impute_unknown(df: pl.DataFrame, col: str, unknown_value: str = "Unknown") -> pl.DataFrame:
    return df.with_columns(
        pl.col(col).fill_null(unknown_value).alias(f"{col}_imputed")
    )
```

### Date Imputation

**Business Logic:** Use related dates.

```python
def impute_dates_business_logic(df: pl.DataFrame) -> pl.DataFrame:
    return df.with_columns(
        # Missing ship_date for shipped orders: use order_date + 1 day
        pl.when(
            pl.col("ship_date").is_null() & pl.col("status").is_in(["shipped", "delivered"])
        )
        .then(pl.col("order_date") + pl.duration(days=1))
        .otherwise(pl.col("ship_date"))
        .alias("ship_date_imputed"),
        
        # Missing delivery_date for delivered orders: use ship_date + 3 days
        pl.when(
            pl.col("delivery_date").is_null() & pl.col("status") == "delivered"
        )
        .then(pl.col("ship_date") + pl.duration(days=3))
        .otherwise(pl.col("delivery_date"))
        .alias("delivery_date_imputed"),
    )
```

### When NOT to Impute

Imputation is not always appropriate:

**Critical fields:** Don't impute primary keys, transaction IDs, or audit-critical data.

**High missing rate:** If 50% is missing, imputation creates more fiction than fact.

**Systematic missingness:** If missingness has meaning (e.g., "not applicable"), imputing destroys information.

**Downstream sensitivity:** If downstream analysis is sensitive to imputed values, leave nulls for explicit handling.

```python
def should_impute(df: pl.DataFrame, col: str, max_missing_rate: float = 0.1) -> bool:
    """Decide if imputation is appropriate."""
    missing_rate = df.select(pl.col(col).null_count()).item() / df.height
    
    if missing_rate > max_missing_rate:
        return False
    
    # Check for systematic patterns (example: all nulls in one category)
    if df.schema.get(col) and "category" in df.columns:
        by_category = df.group_by("category").agg(
            pl.col(col).null_count().alias("nulls"),
            pl.count().alias("total")
        ).with_columns(
            (pl.col("nulls") / pl.col("total")).alias("null_rate")
        )
        
        # If any category is entirely null, missingness is systematic
        if by_category.filter(pl.col("null_rate") == 1.0).height > 0:
            return False
    
    return True
```

---

## 5. Quarantine Patterns

Quarantine separates problematic data for review without losing it.

### When to Quarantine vs Reject vs Fix

**Quarantine when:**
- Data is potentially valuable but needs investigation
- Automatic fix is uncertain
- Human review can recover the data
- You need audit trail of problematic records

**Reject when:**
- Data is clearly garbage
- No possibility of recovery
- Cost of keeping exceeds value
- Processing would cause errors

**Fix when:**
- Correction is deterministic and safe
- Business rules are clear
- Audit requirements allow modification

### Quarantine Table Design

Store quarantined data with context:

```python
def quarantine_records(
    df: pl.DataFrame,
    condition: pl.Expr,
    reason: str
) -> tuple:
    """Separate quarantined records with metadata."""
    quarantined = df.filter(condition).with_columns(
        pl.lit(reason).alias("quarantine_reason"),
        pl.lit(datetime.now()).alias("quarantine_timestamp"),
    )
    
    clean = df.filter(~condition)
    
    return clean, quarantined

# Usage
clean_df, quarantined_df = quarantine_records(
    df,
    pl.col("amount") > 100000,
    "Amount exceeds $100k threshold — manual review required"
)

# Append to quarantine table
quarantined_df.write_parquet("quarantine/high_amounts.parquet", mode="append")
```

### Quarantine Schema

A quarantine table should include:

```python
QUARANTINE_SCHEMA = {
    # Original record (all columns)
    "original_data": pl.Struct,  # Or keep columns flat
    
    # Quarantine metadata
    "quarantine_reason": pl.Utf8,
    "quarantine_rule": pl.Utf8,  # Which validation rule failed
    "quarantine_timestamp": pl.Datetime,
    "source_file": pl.Utf8,
    "batch_id": pl.Utf8,
    
    # Review metadata
    "review_status": pl.Utf8,  # pending, approved, rejected, fixed
    "reviewed_by": pl.Utf8,
    "reviewed_at": pl.Datetime,
    "review_notes": pl.Utf8,
    
    # Resolution
    "resolution": pl.Utf8,  # kept_as_is, corrected, deleted
    "corrected_data": pl.Struct,  # If corrected
}
```

### Review Workflows

Structure the review process:

```python
class QuarantineManager:
    """Manage quarantine review workflow."""
    
    def __init__(self, quarantine_path: str):
        self.path = quarantine_path
    
    def get_pending_review(self, limit: int = 100) -> pl.DataFrame:
        """Get records pending review."""
        return (
            pl.read_parquet(f"{self.path}/*.parquet")
            .filter(pl.col("review_status") == "pending")
            .sort("quarantine_timestamp")
            .head(limit)
        )
    
    def mark_reviewed(
        self,
        record_ids: list,
        status: str,
        reviewer: str,
        notes: str = ""
    ):
        """Update review status for records."""
        df = pl.read_parquet(f"{self.path}/*.parquet")
        
        df = df.with_columns(
            pl.when(pl.col("record_id").is_in(record_ids))
            .then(pl.lit(status))
            .otherwise(pl.col("review_status"))
            .alias("review_status"),
            
            pl.when(pl.col("record_id").is_in(record_ids))
            .then(pl.lit(reviewer))
            .otherwise(pl.col("reviewed_by"))
            .alias("reviewed_by"),
            
            pl.when(pl.col("record_id").is_in(record_ids))
            .then(pl.lit(datetime.now()))
            .otherwise(pl.col("reviewed_at"))
            .alias("reviewed_at"),
        )
        
        df.write_parquet(f"{self.path}/reviewed.parquet")
    
    def reprocess_approved(self) -> pl.DataFrame:
        """Get approved records for reprocessing."""
        return (
            pl.read_parquet(f"{self.path}/*.parquet")
            .filter(pl.col("review_status") == "approved")
            .filter(pl.col("resolution") != "deleted")
        )
```

---

## 6. Rejection and Escalation

Some data should not be processed at all.

### Hard Rejection Criteria

Define what is absolutely unacceptable:

```python
HARD_REJECTION_RULES = [
    {
        "name": "null_primary_key",
        "condition": pl.col("order_id").is_null(),
        "reason": "Cannot process without order ID",
    },
    {
        "name": "future_order_date",
        "condition": pl.col("order_date") > date.today(),
        "reason": "Order date cannot be in future",
    },
    {
        "name": "impossible_amount",
        "condition": pl.col("amount") < 0,
        "reason": "Negative amounts not allowed",
    },
]

def apply_hard_rejections(df: pl.DataFrame, rules: list) -> tuple:
    """Reject records that violate hard rules."""
    rejected = pl.DataFrame()
    
    for rule in rules:
        violations = df.filter(rule["condition"]).with_columns(
            pl.lit(rule["name"]).alias("rejection_rule"),
            pl.lit(rule["reason"]).alias("rejection_reason"),
        )
        rejected = pl.concat([rejected, violations]) if rejected.height > 0 else violations
        df = df.filter(~rule["condition"])
    
    return df, rejected
```

### Escalation to Source Systems

When data quality problems are systemic, escalate:

```python
def generate_escalation_report(
    rejected_df: pl.DataFrame,
    quarantined_df: pl.DataFrame
) -> dict:
    """Generate report for escalation to data producers."""
    return {
        "summary": {
            "rejected_count": rejected_df.height,
            "quarantined_count": quarantined_df.height,
            "report_date": datetime.now().isoformat(),
        },
        "rejection_breakdown": (
            rejected_df
            .group_by("rejection_rule")
            .count()
            .to_dicts()
        ),
        "quarantine_breakdown": (
            quarantined_df
            .group_by("quarantine_reason")
            .count()
            .to_dicts()
        ),
        "sample_rejections": rejected_df.head(10).to_dicts(),
        "sample_quarantine": quarantined_df.head(10).to_dicts(),
    }
```

### Feedback Loops

Create mechanisms for upstream improvement:

```python
def create_feedback_loop_data(df: pl.DataFrame) -> dict:
    """Prepare data for feedback to source systems."""
    return {
        "quality_metrics": {
            "null_rates": {col: df[col].null_count() / df.height for col in df.columns},
            "violation_rates": {},  # Populate from validation results
        },
        "recurring_issues": [],  # Track issues that appear repeatedly
        "recommended_fixes": [],  # Suggest source system changes
        "trend": [],  # Quality over time
    }
```

---

## 7. Correction Audit Trail

Track all corrections for debugging, compliance, and rollback.

### What Was Changed, When, Why

Every correction should be logged:

```python
@dataclass
class CorrectionRecord:
    record_id: str
    column: str
    original_value: any
    corrected_value: any
    correction_type: str  # deterministic, probabilistic, imputation, manual
    correction_rule: str
    timestamp: datetime
    confidence: float = 1.0  # For probabilistic corrections

def log_corrections(
    df_before: pl.DataFrame,
    df_after: pl.DataFrame,
    columns: list,
    correction_type: str,
    correction_rule: str
) -> list:
    """Log all corrections made."""
    records = []
    
    for col in columns:
        # Find rows where values differ
        changed = df_before.join(
            df_after.select(["id", col]).rename({col: f"{col}_after"}),
            on="id"
        ).filter(
            pl.col(col) != pl.col(f"{col}_after")
        )
        
        for row in changed.iter_rows(named=True):
            records.append(CorrectionRecord(
                record_id=row["id"],
                column=col,
                original_value=row[col],
                corrected_value=row[f"{col}_after"],
                correction_type=correction_type,
                correction_rule=correction_rule,
                timestamp=datetime.now(),
            ))
    
    return records
```

### Before/After Snapshots

Maintain snapshots for comparison:

```python
def create_correction_snapshot(
    df_before: pl.DataFrame,
    df_after: pl.DataFrame,
    output_path: str
):
    """Save before/after snapshot for audit."""
    snapshot = {
        "timestamp": datetime.now().isoformat(),
        "before_row_count": df_before.height,
        "after_row_count": df_after.height,
        "columns_modified": [],
    }
    
    # Identify changed columns
    for col in df_before.columns:
        if col in df_after.columns:
            before_vals = set(df_before[col].to_list())
            after_vals = set(df_after[col].to_list())
            if before_vals != after_vals:
                snapshot["columns_modified"].append(col)
    
    # Save
    df_before.write_parquet(f"{output_path}/before.parquet")
    df_after.write_parquet(f"{output_path}/after.parquet")
    
    with open(f"{output_path}/metadata.json", "w") as f:
        json.dump(snapshot, f, indent=2)
```

### Rollback Capability

Design corrections to be reversible:

```python
class ReversibleCorrection:
    """Wrapper for reversible correction operations."""
    
    def __init__(self, df: pl.DataFrame):
        self.original = df.clone()
        self.current = df.clone()
        self.corrections_log = []
    
    def apply(self, correction_func, description: str) -> 'ReversibleCorrection':
        """Apply a correction, keeping rollback capability."""
        before = self.current.clone()
        self.current = correction_func(self.current)
        
        self.corrections_log.append({
            "description": description,
            "timestamp": datetime.now(),
            "rows_before": before.height,
            "rows_after": self.current.height,
        })
        
        return self
    
    def rollback(self, steps: int = 1) -> pl.DataFrame:
        """Rollback to original state."""
        # For simplicity, return to original
        # More sophisticated: checkpoint after each step
        return self.original.clone()
    
    def get_current(self) -> pl.DataFrame:
        return self.current
    
    def get_log(self) -> list:
        return self.corrections_log
```

---

## Summary

Correction strategies balance data quality against risk of over-correction.

**Decision framework:** Can it be fixed? Should it be fixed? What's the risk of fixing it wrong?

**Categories:** Deterministic corrections are safe. Probabilistic corrections need logging. Manual corrections need workflows.

**Imputation:** Match strategy to data type and missingness pattern. Know when not to impute.

**Quarantine:** Preserve problematic data for review rather than discarding silently.

**Audit trail:** Track what changed, when, and why. Enable rollback.

---

## Interview Framing

**"How do you handle data quality issues in a pipeline?"**

"I classify issues by how they can be corrected. Deterministic fixes — like format normalization — are automated. Probabilistic fixes — like typo correction — are logged with confidence scores and reviewed. Issues requiring judgment are quarantined for manual review rather than auto-corrected or silently dropped. Everything is audited: what the original value was, what it was changed to, why, and when. This lets me rollback if corrections were wrong and provides an audit trail for compliance."

**"When would you reject data versus try to fix it?"**

"Reject when the data is clearly invalid and unfixable — null primary keys, logically impossible values, complete garbage. Quarantine when it's potentially valuable but needs human judgment — high-value outliers, ambiguous entries, complex entity resolution. Auto-fix only when the correction is deterministic and safe — format standardization, known typo mappings. The key is being conservative: flag uncertain cases for review rather than making potentially wrong corrections that are harder to undo."
