# Schema Evolution in Data Lake Table Formats

## Summary

Schema evolution allows tables to adapt their structure over time without requiring full data rewrites. As business requirements change, new columns are added, types are widened, and fields are renamed. Table formats handle these changes gracefully, maintaining compatibility between old data files and new schemas.

Key points to remember:

- Schema evolution modifies metadata, not data files, making changes fast and inexpensive
- Backward-compatible changes (adding nullable columns, widening types) are safe and automatic
- Breaking changes (removing columns, narrowing types) require explicit migration
- Column identification methods differ: Delta Lake uses names, Iceberg uses IDs, Hudi uses names
- Iceberg offers the most flexible schema evolution with hidden column IDs
- Schema enforcement prevents accidental corruption by rejecting non-conforming writes
- Old data files are read with the new schema through projection and type coercion

## Why Schema Evolution Matters

Data schemas change constantly in production systems:

- New features require additional columns
- Analytics needs drive column additions
- Data model refinements rename or restructure fields
- Type mismatches are discovered and corrected
- Deprecated fields are removed

Without schema evolution, any schema change would require:
1. Creating a new table with the new schema
2. Copying all existing data (potentially petabytes)
3. Swapping table references in all consumers
4. Deleting the old table

This is slow, expensive, and error-prone. Schema evolution eliminates the data copy by making schema changes metadata-only operations.

## Types of Schema Changes

### Safe Changes (Backward Compatible)

These changes can be applied without data migration:

**Adding nullable columns**: New columns default to NULL for existing rows.

```sql
ALTER TABLE my_table ADD COLUMN new_field STRING
```

**Widening numeric types**: int to long, float to double. Values are automatically promoted.

```sql
ALTER TABLE my_table ALTER COLUMN count TYPE BIGINT
```

**Adding column comments and metadata**: Descriptive changes do not affect data.

### Potentially Breaking Changes

These changes require careful handling:

**Renaming columns**: Depends on whether the format tracks columns by name or ID.
- Delta Lake: Uses column names, so renames require mapping configuration
- Iceberg: Uses column IDs, so renames are metadata-only
- Hudi: Uses column names similar to Delta Lake

**Reordering columns**: Usually safe but may affect consumers that rely on position.

**Changing nullability**: Making a nullable column required needs backfill of NULL values.

### Breaking Changes (Require Migration)

These changes cannot be done in place:

**Removing columns**: May break downstream consumers. Usually done with a deprecation period.

**Narrowing types**: long to int may cause data loss if values exceed the target range.

**Incompatible type changes**: string to int requires data transformation, not just metadata change.

## Column Identity and Tracking

### Name-Based Tracking (Delta Lake, Hudi)

Delta Lake and Hudi identify columns by name. This is simple but creates challenges:

**Rename limitations**: Renaming a column creates a new logical column. Old data files with the original name require mapping.

```python
# Delta Lake column mapping mode
spark.sql("""
  ALTER TABLE my_table SET TBLPROPERTIES (
    'delta.columnMapping.mode' = 'name'
  )
""")
# Now renames work without rewriting data
spark.sql("ALTER TABLE my_table RENAME COLUMN old_name TO new_name")
```

**Case sensitivity**: Column name matching may be case-sensitive or case-insensitive depending on configuration.

### ID-Based Tracking (Iceberg)

Iceberg assigns a unique ID to each column at creation time. The schema stores both the ID and the current name:

```
Column ID: 1, Name: user_id, Type: long
Column ID: 2, Name: email, Type: string
Column ID: 3, Name: created_at, Type: timestamp
```

When reading old Parquet files, Iceberg maps file columns to table columns by ID:
1. Old file has column named "user_id" with ID 1
2. Table schema renames column to "customer_id" but ID remains 1
3. Data from "user_id" in the file maps to "customer_id" in the query

This enables:
- Free column renames (ID stays the same)
- Column additions in any position
- Robust handling of Parquet files with different column orders

## Schema Enforcement

### Write-Time Validation

Table formats enforce schema on write, preventing data corruption:

```python
# This will fail if schema does not match
df.write.format("delta").mode("append").save(path)
# Error: Column 'new_column' is not in the table schema
```

Enforcement catches:
- Missing required columns
- Extra columns not in schema
- Type mismatches
- Nulls in non-nullable columns

### Merge Schema Mode

When adding new columns, explicitly enable schema merging:

**Delta Lake**:
```python
df.write.format("delta") \
    .mode("append") \
    .option("mergeSchema", "true") \
    .save(path)
```

**Iceberg**:
```sql
ALTER TABLE my_table ADD COLUMN new_column STRING
-- Or use merge-schema mode in Spark
```

**Hudi**:
```python
.option("hoodie.datasource.write.reconcile.schema", "true")
```

This is a safety feature: accidental schema changes require explicit opt-in.

### Overwrite Schema Mode

For complete schema replacement (use with caution):

```python
df.write.format("delta") \
    .mode("overwrite") \
    .option("overwriteSchema", "true") \
    .save(path)
```

This replaces both data and schema. Use only when intentionally restructuring a table.

## Implementation by Format

### Delta Lake Schema Evolution

Delta Lake stores the schema in the transaction log as JSON:

```json
{
  "metaData": {
    "schemaString": "{\"type\":\"struct\",\"fields\":[...]}",
    ...
  }
}
```

Schema changes create a new metadata action in the log. Old data files are not modified.

**Supported operations**:
- Add columns (nullable by default)
- Change column comments
- Reorder columns
- Rename columns (with column mapping enabled)
- Widen numeric types
- Change nullability (nullable to not-nullable requires data validation)

**Column mapping modes**:
- `none`: Default, columns matched by name
- `name`: Enables renames without data rewrite
- `id`: Most flexible, similar to Iceberg

```sql
ALTER TABLE my_table SET TBLPROPERTIES (
  'delta.columnMapping.mode' = 'name'
)
```

### Apache Hudi Schema Evolution

Hudi stores schema in Avro format within the table metadata:

```python
# Enable schema evolution
.option("hoodie.schema.on.read.enable", "true")
```

**Supported operations**:
- Add columns at any position
- Rename columns (limited support)
- Widen types (int to long, float to double)
- Change nullability (limited)

Hudi has historically been more restrictive about schema evolution than Iceberg or Delta Lake, but recent versions have improved support.

**Schema reconciliation**:
```python
.option("hoodie.datasource.write.reconcile.schema", "true")
```

This merges the incoming DataFrame schema with the existing table schema.

### Apache Iceberg Schema Evolution

Iceberg provides the most comprehensive schema evolution through its ID-based column tracking:

```sql
-- Add column
ALTER TABLE my_table ADD COLUMN new_col STRING

-- Rename column (metadata only)
ALTER TABLE my_table RENAME COLUMN old_name TO new_name

-- Change column type (compatible widening)
ALTER TABLE my_table ALTER COLUMN count TYPE BIGINT

-- Reorder columns
ALTER TABLE my_table ALTER COLUMN col1 AFTER col2

-- Make column optional
ALTER TABLE my_table ALTER COLUMN required_col DROP NOT NULL

-- Drop column (metadata operation, files unchanged)
ALTER TABLE my_table DROP COLUMN deprecated_col
```

Iceberg tracks schema changes with full history, enabling time travel queries with the schema as it existed at that version.

**Nested schema evolution**: Iceberg supports modifying nested structures within structs, maps, and lists:

```sql
ALTER TABLE my_table ADD COLUMN metadata.source STRING
ALTER TABLE my_table ALTER COLUMN nested.field TYPE BIGINT
```

## Reading Old Data with New Schemas

### Projection and Null Filling

When reading old data files that lack new columns:

```
New schema: [id, name, email, created_at]
Old file:   [id, name]
Result:     [id, name, NULL, NULL]
```

The table format projects available columns and fills missing columns with NULL.

### Type Coercion

When types have widened:

```
New schema: count BIGINT
Old file:   count INT (value: 42)
Result:     count BIGINT (value: 42)
```

Compatible type promotions happen automatically during read.

### Renamed Columns

With ID-based tracking (Iceberg) or name mapping (Delta Lake with column mapping):

```
New schema: customer_id BIGINT (ID: 1)
Old file:   user_id BIGINT (ID: 1)
Result:     customer_id BIGINT (value from user_id)
```

The column ID or mapping ensures data flows to the correct column regardless of name changes.

## Schema Evolution Patterns

### Additive Evolution

The safest pattern adds columns without modifying existing ones:

```sql
-- Version 1
CREATE TABLE orders (id BIGINT, amount DECIMAL(10,2))

-- Version 2: Add shipping
ALTER TABLE orders ADD COLUMN shipping_address STRING

-- Version 3: Add tracking
ALTER TABLE orders ADD COLUMN tracking_number STRING
```

This pattern is always backward compatible. Existing queries continue to work.

### Deprecation Pattern

When removing columns, use a phased approach:

1. Stop writing to the column
2. Mark column as deprecated in documentation
3. Wait for consumers to migrate
4. Drop the column from schema

```sql
-- Phase 1: Add new column, continue writing old
ALTER TABLE users ADD COLUMN email_v2 STRING

-- Phase 2: Write to both during transition
-- Application logic handles migration

-- Phase 3: Drop old column after all consumers migrated
ALTER TABLE users DROP COLUMN email
```

### Wide Table Pattern

For frequently changing schemas, create a wide table with many nullable columns:

```sql
CREATE TABLE events (
  event_id BIGINT,
  event_type STRING,
  timestamp TIMESTAMP,
  attr_string_1 STRING,
  attr_string_2 STRING,
  ...
  attr_int_1 BIGINT,
  attr_int_2 BIGINT,
  ...
)
```

New attributes map to pre-existing columns. This avoids frequent schema changes at the cost of readability.

A better alternative is nested structures:

```sql
CREATE TABLE events (
  event_id BIGINT,
  event_type STRING,
  timestamp TIMESTAMP,
  attributes MAP<STRING, STRING>
)
```

### Struct Evolution

Nested structures can evolve independently:

```sql
-- Original
CREATE TABLE customers (
  id BIGINT,
  address STRUCT<street: STRING, city: STRING>
)

-- Add zip code to address struct
ALTER TABLE customers ADD COLUMN address.zip STRING
```

This is particularly useful for semi-structured data that gains fields over time.

## Comparison Across Formats

| Capability | Delta Lake | Hudi | Iceberg |
|------------|------------|------|---------|
| Add columns | Yes | Yes | Yes |
| Remove columns | Yes | Limited | Yes |
| Rename columns | With mapping | Limited | Yes (free) |
| Reorder columns | Yes | Limited | Yes |
| Widen types | Yes | Yes | Yes |
| Nested evolution | Limited | Limited | Full |
| Column tracking | Name (or ID) | Name | ID |
| Evolution history | Log-based | Timeline | Versioned |

Iceberg leads in schema evolution flexibility due to its ID-based column tracking. Delta Lake is close behind with column mapping enabled. Hudi has historically been more limited but is improving.

## Best Practices

### Design for Evolution

Anticipate schema changes in initial design:
- Use nullable columns by default
- Avoid overly specific column names that may need renaming
- Consider nested structures for complex, evolving data

### Document Schema Changes

Maintain a schema changelog:
- Date and version of each change
- Reason for the change
- Impact on existing consumers
- Migration steps if any

### Test Schema Compatibility

Before applying changes to production:
1. Apply the change to a development copy
2. Query old data with new schema
3. Query new data with both old and new reader code
4. Validate that transformations work correctly

### Coordinate with Consumers

Schema changes affect downstream systems:
- Notify data consumers before breaking changes
- Provide migration windows for deprecated columns
- Maintain backward compatibility when possible

### Use Schema Registry for Streaming

For streaming ingestion, integrate with a schema registry:
- Confluent Schema Registry for Kafka
- AWS Glue Schema Registry
- Custom registry solutions

This centralizes schema management and enforces compatibility rules before data reaches the lake.
