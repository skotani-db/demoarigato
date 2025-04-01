# Databricks notebook source
# Install dependencies
%pip install pandas numpy faker

# COMMAND ----------

# Import packages
import re
import pandas as pd
import numpy as np
import json
import datetime
from faker import Faker

# Get notebook parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("synthetic_table", "")
dbutils.widgets.text("schema_distribution", "")
dbutils.widgets.text("num_records", "50")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
synthetic_table = dbutils.widgets.get("synthetic_table")
schema_distribution = dbutils.widgets.get("schema_distribution")
num_records = int(dbutils.widgets.get("num_records"))

fake = Faker()

def clean_json_random_expressions(json_str):
    """
    Convert random value expressions in JSON to actual values.
    
    Parameters
    ----------
    json_str : str
        JSON string to be cleaned
    
    Returns
    -------
    dict
        Dictionary with evaluated values
    """
    # Parse JSON
    data_dict = json.loads(json_str)
    
    # Create safe evaluation environment
    safe_globals = {
        "np": np,
        "fake": fake,
        "round": round
    }
    
    # Evaluate expressions for each key
    for key, value in data_dict.items():
        try:
            # Evaluate expressions stored as strings
            data_dict[key] = eval(value, {"__builtins__": {}}, safe_globals)
        except Exception as e:
            print(f"Error evaluating expression for {key}: {e}")
            # Keep original value on error
    
    return data_dict

def generate_dataframe_from_json(schema_distribution, num_records=50):
    """
    Generate DataFrame from JSON schema specification.
    
    Parameters
    ----------
    schema_distribution : str
        JSON string containing column specifications
    num_records : int, default=50
        Number of records to generate
    
    Returns
    -------
    pandas.DataFrame
        DataFrame with generated data
    """
    # Clean JSON string
    schema_distribution = schema_distribution.replace("```json", "").replace("```python", "").replace("```", "")

    rows = []
    for i in range(num_records):
        row = clean_json_random_expressions(schema_distribution)
        rows.append(row)
    
    # Convert to DataFrame
    df = pd.DataFrame(rows)
    
    return df

# COMMAND ----------

# Initialize - create only the base synthetic data table
spark.sql(f"DROP TABLE IF EXISTS {catalog}.{schema}.{synthetic_table}")

# Generate synthetic data
print(f"Generating {num_records} records for {synthetic_table}")
synthetic_df = generate_dataframe_from_json(schema_distribution, num_records=num_records)
synthetic_spark_df = spark.createDataFrame(synthetic_df)

# Save synthetic data to table
synthetic_spark_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{synthetic_table}")

# Display sample records
df = spark.sql(f"SELECT * FROM {catalog}.{schema}.{synthetic_table} LIMIT 3")
display(df)

print(f"Created table {catalog}.{schema}.{synthetic_table} with {num_records} records")

# Return the result as dictionary with table information
result = {
    "table": f"{catalog}.{schema}.{synthetic_table}", 
    "num_records": num_records,
    "columns": synthetic_df.columns.tolist()
}

# COMMAND ----------

