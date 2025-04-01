# Databricks notebook source
# Install dependencies
%pip install --upgrade databricks-sdk
%restart_python

# COMMAND ----------

# Databricks notebook source
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

w = WorkspaceClient()

# Get notebook parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("synthetic_table", "")
dbutils.widgets.text("source_schema", "")
dbutils.widgets.text("warehouse_id", "")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
synthetic_table = dbutils.widgets.get("synthetic_table")
source_schema = dbutils.widgets.get("source_schema")
warehouse_id = dbutils.widgets.get("warehouse_id")

# Parse source_schema
source_schema_dict = json.loads(source_schema.replace("```json", "").replace("```python", "").replace("```", ""))

def get_common_columns(schema_dict):
    """
    Get columns common to all tables in the schema.
    
    Args:
        schema_dict: Dictionary with table definitions
        
    Returns:
        List of common column names
    """
    # Get all table columns as sets
    table_columns = [set(table_info["columns"]) for table_info in schema_dict.values()]
    
    # Find intersection of all sets
    common_columns = table_columns[0]
    for columns in table_columns[1:]:
        common_columns = common_columns.intersection(columns)
    
    return list(common_columns)

def create_raw_tables(schema_dict):
    """
    Create raw tables from the synthetic data based on schema definition.
    
    Args:
        schema_dict: Dictionary with table definitions
        
    Returns:
        List of created table names
    """
    created_table_names = []
    
    # Load the synthetic data
    synthetic_df = spark.table(f"{catalog}.{schema}.{synthetic_table}")
    
    # Get tables and columns from schema
    raw_tables = schema_dict.keys()
    
    for table_name in raw_tables:
        # Get columns for this table
        columns = schema_dict[table_name]["columns"]
        
        # Raw table name
        raw_table_name = f"raw_{table_name}"
        
        # Drop if exists
        spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{raw_table_name}")
        
        # Create new table with selected columns
        synthetic_df.select(*columns).write.mode("overwrite").saveAsTable(
            f"{catalog}.{schema}.{raw_table_name}"
        )
        
        created_table_names.append(raw_table_name)
        print(f"Created table {catalog}.{schema}.{raw_table_name}")
    
    return created_table_names

def create_silver_tables(raw_table_names):
    """
    Create silver tables from raw tables using SQL.
    
    Args:
        raw_table_names: List of raw table names
        
    Returns:
        List of created silver table names
    """
    created_table_names = []
    created_sql_ids = []
    
    for raw_name in raw_table_names:
        # Silver table name
        silver_name = raw_name.replace("raw_", "silver_")
        
        # Drop if exists
        spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{silver_name}")
        
        # Create silver table
        ddl = f"""
        CREATE OR REPLACE TABLE {catalog}.{schema}.{silver_name} AS 
        SELECT * FROM {catalog}.{schema}.{raw_name}
        """
        spark.sql(ddl)

        # DDLをクエリに追加
        query = w.queries.create(
            query=sql.CreateQueryRequestQuery(
                display_name=f"create_{silver_name}",
                warehouse_id=warehouse_id,
                description="query for {silver_name}",
                query_text=ddl,
            )
        )

        created_table_names.append(silver_name)
        created_sql_ids.append(query.id)
    
    return created_table_names, created_sql_ids

def create_gold_table(silver_table_names, schema_dict):
    """
    Create gold table by joining silver tables.
    
    Args:
        silver_table_names: List of silver table names
        schema_dict: Dictionary with table definitions
        
    Returns:
        Name of created gold table
    """
    # Get join keys
    join_keys = get_common_columns(schema_dict)
    
    # Build join condition and except columns
    join_condition = " AND ".join([f"t1.{key}=t2.{key}" for key in join_keys])
    except_columns = ", ".join([f"t2.{key}" for key in join_keys])
    
    # Drop if exists
    spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{synthetic_table}")
    
    # Create gold table by joining silver tables
    ddl = f"""
    CREATE OR REPLACE TABLE {catalog}.{schema}.{synthetic_table} AS 
    SELECT * EXCEPT({except_columns}) 
    FROM {catalog}.{schema}.{silver_table_names[0]} t1 
    JOIN {catalog}.{schema}.{silver_table_names[1]} t2 
    ON {join_condition}
    """
    spark.sql(ddl)

    # DDLをクエリに追加
    query = w.queries.create(
        query=sql.CreateQueryRequestQuery(
            display_name=f"create_{synthetic_table}",
            warehouse_id=warehouse_id,
            description="query for {synthetic_table}",
            query_text=ddl,
        )
    )
    
    return synthetic_table, query.id

# COMMAND ----------

# Execute requested operation
result = {}
print(source_schema_dict)
# Create raw tables
raw_table_names = create_raw_tables(source_schema_dict)
result["raw_table_names"] = raw_table_names

# Get raw table names
# raw_table_names = [f"raw_{table}" for table in source_schema_dict.keys()]

# Create silver tables
silver_table_names, silver_table_queries = create_silver_tables(raw_table_names)
result["silver_table_names"] = silver_table_names
result["silver_table_queries"] = silver_table_queries

# Get silver table names
# silver_table_names = [f"silver_{table}" for table in source_schema_dict.keys()]

# Create gold table
gold_table_name, gold_table_query = create_gold_table(silver_table_names, source_schema_dict)
result["gold_table_name"] = gold_table_name
result["gold_table_query"] = gold_table_query

# Return the result
dbutils.notebook.exit(json.dumps(result))