import os
import json
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql
from databricks.sdk.service.jobs import JobSettings
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
from langchain import PromptTemplate

class DatabricksManager:
    """
    Manages interactions with Databricks, providing a unified interface for jobs, endpoints, and queries.
    """
    
    def __init__(self, config_file="config.json"):
        """
        Initialize the Databricks manager with configuration.
        
        Args:
            config_file (str): Path to JSON configuration file
        """
        # Load configuration
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            self.server_hostname = config.get('SERVER_HOSTNAME')
            self.databricks_host = config.get('DATABRICKS_HOST', f"https://{self.server_hostname}/")
            self.endpoint_name = config.get('ENDPOINT_NAME')
            self.catalog = config.get('CATALOG')
            self.schema = config.get('SCHEMA')
            self.warehouse_id = config.get('WAREHOUSE_ID')
            self.cluster_id = config.get('CLUSTER_ID')
            self.user_name = config.get('USER_NAME')
            self.root_dir = config.get('ROOT_DIR')
        else:
            # Fall back to environment variables
            self.server_hostname = os.environ.get('DATABRICKS_SERVER_HOSTNAME')
            self.databricks_host = os.environ.get('DATABRICKS_HOST', f"https://{self.server_hostname}/")
            self.endpoint_name = os.environ.get('DATABRICKS_ENDPOINT_NAME')
            self.catalog = os.environ.get('DATABRICKS_CATALOG', 'main')
            self.schema = os.environ.get('DATABRICKS_SCHEMA', 'default')
            self.warehouse_id = os.environ.get('DATABRICKS_WAREHOUSE_ID')
            self.cluster_id = os.environ.get('DATABRICKS_CLUSTER_ID')
            self.user_name = config.get('USER_NAME')
            self.root_dir = config.get('ROOT_DIR')
        
        # Initialize Databricks client
        self.client = WorkspaceClient()
    
    def call_model_endpoint(self, messages, max_tokens=5028):
        """
        Call a model endpoint with the specified messages.
        
        Args:
            messages (list): List of messages to send to the model
            max_tokens (int): Maximum tokens in the response
            
        Returns:
            str: Model response content
        """
        chat_messages = [
            ChatMessage(
                content=message["content"],
                role=ChatMessageRole[message["role"].upper()]
            ) if isinstance(message, dict) else ChatMessage(
                content=message, 
                role=ChatMessageRole.USER
            )
            for message in messages
        ]
        
        response = self.client.serving_endpoints.query(
            name=self.endpoint_name,
            messages=chat_messages,
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    
    def run_chain(self, prompt_template, max_tokens=5028, **kwargs):
        """
        Format a prompt and run it through the model endpoint.
        
        Args:
            prompt_template: PromptTemplate or string template
            max_tokens (int): Maximum tokens in the response
            **kwargs: Variables to format the template
            
        Returns:
            str: Model response
        """
        if isinstance(prompt_template, PromptTemplate):
            formatted_prompt = prompt_template.format(**kwargs)
        else:
            formatted_prompt = prompt_template.format(**kwargs)
        
        messages = [
            {"role": "assistant", "content": "You are a helpful assistant."},
            {"role": "user", "content": formatted_prompt}
        ]
        
        return self.call_model_endpoint(messages, max_tokens=max_tokens)
    
    def create_job(self, job_config):
        """
        Create a new job from a configuration object.
        
        Args:
            job_config (dict): Job configuration
            
        Returns:
            job: Created job object
        """
        job_settings = JobSettings.from_dict(job_config)
        return self.client.jobs.create(**job_settings.as_shallow_dict())
    
    def get_run_output(self, run_id):
        """
        Get the output of a job run.

        Args:
            run_id (str): ID of the run to get

        Returns:
            dict: Output of the job run
        """
        run_output = self.client.jobs.get_run_output(run_id=run_id.tasks[0].run_id)
        return run_output.notebook_output.result
    
    def run_job(self, job_id):
        """
        Run a job by ID.
        
        Args:
            job_id (str): ID of the job to run
            
        Returns:
            run: Job run object
        """
        return self.client.jobs.run_now(job_id=job_id)
    
    def get_experiment_by_name(self, experiment_name):
        """
        Get an experiment by name.

        Args:
            experiment_name (str): Name of the experiment to get
            
        Returns:
            experiment: Experiment object
        """
        return self.client.experiments.get_by_name(experiment_name)
    
    def get_best_run_id(self, experiment_id):
        """
        Get the best run ID for an experiment based on validation R2 score.

        Args:
            experiment_id (str): ID of the experiment

        Returns:
            str: Best run ID
        """
        runs = self.client.experiments.search_runs(experiment_ids=[experiment_id], order_by=["metrics.val_r2_score DESC"])
        run_id_list = [r.info.run_id for r in runs]
        return run_id_list[0]
    
    def get_best_run_notebook(self, experiment_response):
        """
        Get the best run notebook path for an experiment.

        Args:
            experiment_response: Experiment object

        Returns:
            str: Best run notebook path
        """
        def _get_value_by_key(tags_list, target_key):
            for tag in tags_list:
                if tag.key == target_key:
                    return tag.value
            return None
        
        return _get_value_by_key(experiment_response.experiment.tags, "_databricks_automl.best_trial_notebook_path")

    def get_run(self, run_id):
        """
        Get details of a job run.
        
        Args:
            run_id (str): ID of the run
            
        Returns:
            run: Run object with full details
        """
        return self.client.jobs.get_run(run_id=run_id)
    
    def get_run_status(self, run_info):
        """
        Extract the status state from a run info object.
        
        Args:
            run_info: Run object
            
        Returns:
            str: Run status state value
        """
        return run_info.status.state.value
    
    def create_sql_query(self, name, query_text, description=""):
        """
        Create a SQL query in Databricks SQL.
        
        Args:
            name (str): Display name for the query
            query_text (str): SQL query text
            description (str): Description for the query
            
        Returns:
            query: Created query object
        """
        query = self.client.queries.create(
            query=sql.CreateQueryRequestQuery(
                display_name=name,
                warehouse_id=self.warehouse_id,
                description=description,
                query_text=query_text,
            )
        )
        return query
    
    def get_table(self, table_name):
        """
        Get data from a table as a pandas DataFrame.
        
        Args:
            table_name (str): Name of the table (without catalog/schema)
            
        Returns:
            pandas.DataFrame: Table data
        """
        response = self.client.statement_execution.execute_statement(
            statement=f"SELECT * FROM {self.catalog}.{self.schema}.{table_name}",
            warehouse_id=self.warehouse_id
            )
        return pd.DataFrame(response.result.data_array, columns=[c.name for c in response.manifest.schema.columns])
    
    def get_column_types(self, table_name, column_names=None):
        """
        Get the data types of columns from a Unity Catalog table.
        
        Args:
            table_name (str): Name of the table (without catalog/schema)
            column_names (list, optional): List of column names to get types for.
                                        If None, returns types for all columns.
        
        Returns:
            dict: Dictionary mapping column names to their data types
        """
        # Construct the full table name using class variables
        full_table_name = f"{self.catalog}.{self.schema}.{table_name}"
        
        # Use Databricks table API to get schema information directly
        table_info = self.client.tables.get(
            full_name=full_table_name
        )
        
        # Extract column information from the schema
        column_types = {}
        for column in table_info.schema:
            if column_names is None or column.name in column_names:
                column_types[column.name] = column.type_text
    
        return column_types
    
    def update_job_config(self, sql_queries=None, warehouse_id=None, notebook_path=None, cluster_id=None, output_path="updated_pipeline.json"):
        """
        Dynamically update Databricks job configuration with new parameters.
        
        Args:
            sql_queries (dict, optional): Dictionary mapping task keys to query IDs
            warehouse_id (str, optional): New warehouse ID for SQL tasks
            notebook_path (str, optional): New notebook path for training task
            cluster_id (str, optional): New cluster ID for training task
            output_path (str, optional): Path to save updated configuration
            
        Returns:
            dict: Updated job configuration
        """
        # Load the base configuration
        base_path = "pipeline_base.json"
        if not os.path.exists(base_path):
            raise FileNotFoundError(f"Base configuration file {base_path} not found")
        
        with open(base_path, 'r') as f:
            job_config = json.load(f)
        
        # Update SQL query names if provided
        if sql_queries is not None:
            for task in job_config["tasks"]:
                if task["task_key"] in sql_queries and "sql_task" in task:
                    task["sql_task"]["query"]["query_id"] = sql_queries[task["task_key"]]
        
        # Update warehouse ID if provided
        if warehouse_id is not None:
            for task in job_config["tasks"]:
                if "sql_task" in task:
                    task["sql_task"]["warehouse_id"] = warehouse_id
        
        # Update notebook path if provided
        if notebook_path is not None:
            for task in job_config["tasks"]:
                if task["task_key"] == "training_model" and "notebook_task" in task:
                    task["notebook_task"]["notebook_path"] = notebook_path
        
        # Update cluster ID if provided
        if cluster_id is not None:
            for task in job_config["tasks"]:
                if task["task_key"] == "training_model" and "existing_cluster_id" in task:
                    task["existing_cluster_id"] = cluster_id
        
        # Save the updated configuration if output path is provided
        if output_path:
            with open(output_path, 'w') as f:
                json.dump(job_config, f, indent=2)
        
        return job_config