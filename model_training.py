# Databricks notebook source
# Install dependencies
%pip install pandas numpy mlflow

# COMMAND ----------

# Import packages
import pandas as pd
import numpy as np
from databricks import automl
import mlflow
import json

# Get notebook parameters
dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "default")
dbutils.widgets.text("user_name", "")
dbutils.widgets.text("table_name", "")
dbutils.widgets.text("experiment_name", "")
dbutils.widgets.text("model_name", "")
dbutils.widgets.text("target_column", "")
dbutils.widgets.text("problem_type", "regress")

catalog = dbutils.widgets.get("catalog")
schema = dbutils.widgets.get("schema")
user_name = dbutils.widgets.get("user_name")
table_name = dbutils.widgets.get("table_name")
experiment_name = dbutils.widgets.get("experiment_name")
model_name = dbutils.widgets.get("model_name")
target_column = dbutils.widgets.get("target_column")
problem_type = dbutils.widgets.get("problem_type")

def train_model(df, target_column, problem_type, experiment_name, user_name):
    """
    Train a model using AutoML.

    Args:
        df: DataFrame containing the training data
        target_column: Column name to predict
        problem_type: Type of problem (regress, classify)

    Returns:
        AutoML summary object
    """
    if problem_type == "regress":
        return automl.regress(df, target_col=target_column, timeout_minutes=5, experiment_name=experiment_name, experiment_dir="/demoarigato/databricks_automl/")
    elif problem_type == "classify":
        return automl.classify(df, target_col=target_column, timeout_minutes=5, experiment_name=experiment_name, experiment_dir="/demoarigato/databricks_automl/")
    else:
        raise ValueError(f"Unsupported problem type: {problem_type}")

def register_model_to_uc(run_id, model_name, catalog, schema):
    """
    Register a model to Unity Catalog.

    Args:
        run_id: MLflow run ID
        problem_type: Type of problem
        target_column: Target column name
        catalog: Unity Catalog name
        schema: Schema name

    Returns:
        Registered model details
    """
    mlflow.set_registry_uri("databricks-uc")
    model_uri = f"runs:/{run_id}/model"
    model_name = f"{model_name}"

    registered_model_name = f"{catalog}.{schema}.{model_name}"

    model_details = mlflow.register_model(model_uri, registered_model_name)
    return model_details

# Load the data
print(f"Loading data from {catalog}.{schema}.{table_name}")
spark_df = spark.table(f"{catalog}.{schema}.{table_name}")
df = spark_df.toPandas()

# Check if target column exists and is numeric
if target_column not in df.columns:
    raise ValueError(f"Target column '{target_column}' not found in dataset")

if problem_type == "regress" and not np.issubdtype(df[target_column].dtype, np.number):
    raise ValueError(f"Target column '{target_column}' must be numeric for regression")

# Enable MLflow autologging
mlflow.autolog()

# Train the model
print(f"Training {problem_type} model for {target_column}")
summary = train_model(df, target_column, problem_type, experiment_name, user_name)

# Get run information
run_id = summary.best_trial.mlflow_run_id
experiment_id = summary.experiment.experiment_id
best_notebook_path = summary.best_trial.notebook_path
best_notebook_url = summary.best_trial.notebook_url

# Register the model to Unity Catalog
print(f"Registering model to {catalog}.{schema}")
model_details = register_model_to_uc(run_id, model_name, catalog, schema)

# Create result object
result = {
    "run_id": run_id,
    "experiment_id": experiment_id,
    "model_name": model_details.name,
    "model_version": model_details.version,
    "best_notebook_path": best_notebook_path,
    "best_notebook_url": best_notebook_url,
    "best_metrics": {k: float(v) if isinstance(v, np.float32) else v 
                    for k, v in summary.best_trial.metrics.items()}
}

# Return the result as JSON
print("Model training complete")
dbutils.notebook.exit(json.dumps(result))