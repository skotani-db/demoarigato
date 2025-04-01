import streamlit as st
import json
import pandas as pd
import os
import time
from databricks_manager import DatabricksManager
from langchain import PromptTemplate
import random, string

def randomname(n):
   # Generate a random string of specified length
   randlst = [random.choice(string.ascii_letters + string.digits) for i in range(n)]
   return ''.join(randlst)

# Set page configuration
st.set_page_config(page_title="Databricks DemoArigato メイドインジャパン", layout="wide")

# Initialize Databricks Manager
@st.cache_resource
def get_dbx_manager():
    """Initialize and cache the Databricks Manager instance"""
    return DatabricksManager(config_file="config.json")

dbx = get_dbx_manager()

# Application title and description
st.title("Databricks DemoArigato メイドインジャパン")
st.write("Databricks上でユースケースをエンドツーエンドで実装します")

# Sidebar navigation
with st.sidebar:
    st.header("ナビゲーション")
    page = st.radio(
        "ページを選択:",
        ["ビジネス課題", "データスキーマ", "データ生成", "AI/BI",
         "モデル学習", "ジョブ実行", "デモスクリプト"]
    )
    
    # Display environment information
    st.divider()
    st.caption("環境情報")
    st.caption(f"カタログ: {dbx.catalog}")
    st.caption(f"スキーマ: {dbx.schema}")

# Business Challenge page
if page == "ビジネス課題":
    st.header("ビジネス課題に基づいたストーリーの生成")
    
    business_question = st.text_input("ビジネス課題", "おにぎりの需要予測")
    st.session_state.business_question = business_question
    if st.button("ストーリーラインを生成"):
        with st.spinner("ストーリーラインを生成中..."):
            # Define prompt template for use case generation
            use_case_prompt = PromptTemplate(
            input_variables=["business_question"],
            template="{business_question}に関して助けとなるプロンプトが必要です。その業界の従業員だけが知っているような専門的な知識を使用してください。例を1つだけ挙げてください。その例は、後で生成するデータセットのデータ分析を通じて解決できるものにしてください。業界特有の用語を含めてください。ここではまだサンプルデータを作成する必要はありません。データセットを構築するなど、フォローアップの質問はしないでください。"
            )

            # Run the chain to generate use case
            use_case = dbx.run_chain(use_case_prompt, business_question=business_question)
            st.session_state.use_case = use_case

            if "use_case" in st.session_state:
                # Define prompt template for storyline generation
                storyline_prompt = PromptTemplate(
                    input_variables=["use_case"],
                    template="{use_case}の文脈に基づいたストーリーを作成してください。実際の業務で使用するため、プロフェッショナルな内容にしてください。重要な情報のみを含めた、短く簡潔なものにしてください。ここではまだサンプルデータを作成する必要はありません。それについて質問もしないでください。フォローアップの質問もしないでください。"
                )
                
                # Generate storyline from use case
                storyline = dbx.run_chain(storyline_prompt, use_case=st.session_state.use_case)
                st.session_state.storyline = storyline
            
            st.subheader("生成されたコンテンツ")
            
    if "storyline" in st.session_state:
        st.markdown("### ストーリーライン")
        st.write(st.session_state.storyline)

# Data Schema page
elif page == "データスキーマ":
    st.header("データスキーマの生成")
    
    if "use_case" not in st.session_state:
        st.warning("最初にビジネス課題とユースケースを定義してください。")
    else:
        if st.button("スキーマを生成"):
            with st.spinner("スキーマを生成中..."):
                # Define prompt template for general schema generation
                general_schema_prompt = PromptTemplate(
                    input_variables=["use_case"],
                    template="{use_case}に基づいて、それを機械学習もしくはBIで解決するために必要な情報を持った一般的なテーブルを1つ作成してください。テーブルの列数は7-10カラムに制限し、全ての列のタイトルは、アルファベットを用いて、2語以上から構成される場合、スペースの代わりに_をつけてください。_以外の特殊文字は使用できません。スラッシュ(/)は使用不可です。実務家が見るような一般的なフィールドと、実務家しか知らないようなフィールドを混ぜてください。"
                )
                
                # Generate general schema
                general_schema = dbx.run_chain(general_schema_prompt, use_case=st.session_state.use_case)
                st.session_state.general_schema = general_schema

                # Example JSON format for schema definition
                json_example = '{"table_name_1": {"columns": ["client_id", "sales_amount"], "description": "顧客ごとの売上合計"}, "table_name_2": {"columns": ["client_id","transaction_id", "transaction_date", "transaction_amount"], "description": "顧客ごとのトランザクション"}}'
                
                while True:
                    # Define prompt template for source schema generation
                    source_schema_prompt = PromptTemplate(
                        input_variables=["general_schema", "json_example"],
                        template="""{general_schema}を実務上意味のある2つのテーブルへ分割する場合、それぞれのテーブルはどのような意味をもち、どの列を含めるべきかを説明してください。企業内のシステムソースの種類に応じてテーブルデータを分割する。以下のようなJSONの返答のみが必要です。{json_example}。table_nameやcolumnsは、それぞれのテーブルの名前とそのカラムのリストを表します。descriptionは、テーブルの目的や重要なカラムの意味を説明するための短い説明です。カラム名、テーブル名はアルファベットで定義すること。分割の際の結合キーとなる列は1つ選択すること。"""
                    )
                    source_schema = dbx.run_chain(source_schema_prompt, general_schema=st.session_state.general_schema, json_example=json_example)

                    # Validate the generated schema
                    schema_validation_prompt = PromptTemplate(
                        input_variables=["general_schema", "source_schema"],
                        template="""{general_schema}で定義したテーブルを分割して、{source_schema}で定義された2つのテーブルを作成した。この時、{source_schema}が持つ列と{general_schema}の列は互いに全ての列を含んでいるか確認してください。全ての列が含まれている場合はYES, {source_schema}が{general_schema}にない列を含んでいる場合はNOを返してください。YESかNOかで回答し、他の情報や追加の質問はしないでください。"""
                    )
                    schema_validation = dbx.run_chain(schema_validation_prompt, general_schema=st.session_state.general_schema, source_schema=source_schema)

                    if schema_validation == "YES":
                        break

                st.session_state.source_schema = source_schema
                st.session_state.schema_validation = True
        
        st.subheader("生成されたスキーマ")
        
        if "general_schema" in st.session_state:
            st.markdown("### 分析用データのスキーマ")
            st.write(st.session_state.general_schema)
        
        if "source_schema" in st.session_state:
            st.markdown("### データソースのスキーマ")
            st.write(st.session_state.source_schema)
            
            # Display in a more readable format
            try:
                schema_dict = json.loads(st.session_state.source_schema.replace("```json", "").replace("```", ""))
                st.json(schema_dict)
            except:
                pass

# Data Generation page
elif page == "データ生成":
    st.header("合成データの生成")
    if "general_schema" not in st.session_state:
        st.warning("先にスキーマを生成してください。")
    else:
        # Data generation section
        st.subheader("合成データテーブルの作成")
        col1, col2 = st.columns(2)    
        with col1:
            synthetic_table = st.text_input("テーブル名", "synthetic_data")
        
        with col2:
            num_records = st.number_input("レコード数", min_value=10, max_value=1000, value=50)
    
        if st.button("データを生成"):
            # First determine potential problem type for the business challenge
            with st.spinner("データを分析中..."):
                if "problem_type_analyzed" not in st.session_state:
                    # Define prompt template for ML problem type determination
                    ml_problem_prompt = PromptTemplate(
                        input_variables=["business_question"],
                        template="{business_question}に基づいて、実行する機械学習の問題のタイプを以下の選択肢から選びなさい。選択肢は、['regress', 'classify' 'others']です。機械学習モデルを訓練せずに解決できるタイプの問題については、'others'と回答しなさい。提供された選択肢から一つ選択して回答し、それ以外の詳細は作成しないでください。"
                    )
                    
                    problem_type = dbx.run_chain(ml_problem_prompt, business_question=st.session_state.get("business_question", ""))
                    st.session_state.problem_type = problem_type  # Save for model training section
                    
                    # Define prompt template for target column selection
                    target_column_prompt = PromptTemplate(
                        input_variables=["business_question", "problem_type", "general_schema"],
                        template="{business_question}を解決するための、{problem_type}の機械学習のモデル学習における目的変数となるカラムをただ1つだけ選びなさい。提供されたスキーマのカラムに基づいてカラム名のみを回答し、それ以外の詳細は作成しないでください。スキーマ: {general_schema}"
                    )
                    
                    target_column = dbx.run_chain(
                        target_column_prompt, 
                        business_question=st.session_state.get("business_question", ""),
                        problem_type=problem_type,
                        general_schema=st.session_state.get("general_schema", "")
                    )

                    # Save target column for later use
                    st.session_state.target_column = target_column
                    st.session_state.suggested_target = target_column

                    st.session_state.problem_type_analyzed = True
            while True:
                # Generate data distribution
                with st.spinner("データ分布を生成中..."):
                    problem_type = st.session_state.get("problem_type", "unknown")
                    target_column = st.session_state.get("target_column", "unknown")
                    # Define prompt template for schema distribution generation
                    schema_distribution_prompt = PromptTemplate(
                        input_variables=["general_schema", "problem_type", "target_column"],
                        template="{general_schema}に基づいて、各カラムの分布を作成してください。あなたはデータセットの分布の作成には、fakerライブラリとnumpyライブラリのみを使用できます。lambda funtionによる関数化は認めません。あなたが生成するべきものは、データセットの列をキーとして、対応する分布を持ったdictです。私はJSONの返答のみを必要としています。その他の詳細、その他の新しい線やスペースは必要としていません。以下はデータセットの分布の出力例ですので、よく参考にしてください。： \
                        \
                        {{'transaction_id': fake.uuid4(), 'customer_name': fake.name(), 'transaction_date': fake.date_this_year(), 'fraudulent': np.random.binomial(n=1, p=0.02), 'transaction_amount': round(np.random.lognormal(mean=5, sigma=1.5), 2)}}"
                    )
                    schema_distribution = dbx.run_chain(schema_distribution_prompt, 
                                                        general_schema=st.session_state.general_schema,
                                                        problem_type=st.session_state.problem_type,
                                                        target_column=st.session_state.target_column)
                    st.session_state.schema_distribution = schema_distribution

                # Create job configuration
                with st.spinner("ジョブを作成中..."):
                    job_config = {
                        "name": f"{synthetic_table}の生成",
                        "tasks": [{
                            "task_key": "generate_data",
                            "notebook_task": {
                                "notebook_path": dbx.root_dir+"/data_generator",
                                "base_parameters": {
                                    "catalog": dbx.catalog,
                                    "schema": dbx.schema,
                                    "synthetic_table": synthetic_table,
                                    "schema_distribution": st.session_state.schema_distribution,
                                    "num_records": str(num_records)
                                }
                            },
                            "existing_cluster_id": dbx.cluster_id
                        }]
                    }
                    
                    # Create job
                    job = dbx.create_job(job_config)
                    job_id = job.job_id
                    
                    # Run job
                    run = dbx.run_job(job_id)
                    run_id = run.run_id
                    
                    # Get run information
                    run_info = dbx.get_run(run_id)
                    current_status = dbx.get_run_status(run_info)

                    # Save information for later use
                    st.session_state.generate_data_job_id = job_id
                    st.session_state.generate_data_run_id = run_id

                    # Parse URLs
                    job_url = f"{dbx.databricks_host}jobs/{job_id}"
                    run_url = run_info.run_page_url

                    # Display job and run URLs
                    st.markdown(f"{synthetic_table}テーブルに{num_records}件のレコードを生成するジョブを作成して開始しました")
                    # st.markdown(f"- [Databricksでジョブを表示]({job_url})")
                    # st.markdown(f"- [Databricksで実行を表示]({run_url})")

                # Wait for job completion
                with st.spinner("ジョブ実行中..."):
                    while True:
                        time.sleep(5)
                        run_info = dbx.get_run(run_id)
                        current_status = dbx.get_run_status(run_info)
                        if current_status == "TERMINATED":
                            break
                    st.success("ジョブが完了しました")
                    
                # Save table name for later use
                st.session_state.synthetic_table = synthetic_table
                st.session_state.synthetic_table_url = f"{dbx.databricks_host}explore/data/{dbx.catalog}/{dbx.schema}/{synthetic_table}"

                try:
                    df = dbx.get_table(synthetic_table)
                    st.dataframe(df)
                    break
                except Exception as e:
                    # st.error(f"エラー: {e}。再試行してください。")
                    pass

            if "synthetic_table" in st.session_state:
                st.markdown(f"- [Databricksでテーブルを表示]({st.session_state.synthetic_table_url}) ")

# AI/BI page
elif page == "AI/BI":
    st.header("AI/BI Genie/Dashboardのサンプルプロンプトを生成")
    if "general_schema" in st.session_state and st.button("AI/BIのサンプルプロンプトを生成"):
        with st.spinner("AI/BIのサンプルプロンプトを生成中..."):
            # Define prompt template for genie data questions
            genie_data_prompt = PromptTemplate(
                input_variables=["general_schema"],
                template="text-to-sqlのデモを行います。実務家が質問する例を教えてください。最大10個まで。余分なテキストは使わず、質問そのものを生成してください。探索的/要約的な質問から始まり、異常値までドリルダウンされ、その後、別のプロパティによってブレイクアウトされる可能性があることを考慮してください。これはあなたが作業しなければならないデータスキーマです： {general_schema}です。"
            )
            
            genie_questions = dbx.run_chain(genie_data_prompt, general_schema=st.session_state.general_schema)
            st.session_state.genie_questions = genie_questions

            # Define prompt template for lakeview visualization tips
            lakeview_prompt = PromptTemplate(
                input_variables=["genie_questions", "general_schema"],
                template="{genie_questions}からの質問に基づき、一般ユーザー向けにどのような4つの簡単なグラフを作ればよいでしょうか？日本語からビジュアライズするツールにコピー＆ペーストできる形式で提供してください。例：（日本語から視覚化するテキスト： Y軸にclient_id、X軸にsales_amountの棒グラフ) ジオロケーションマップを必要とするものは含めないでください。このスキーマに基づいてのみグラフを作成できます： {general_schema}。各軸に指定するカラム名はスキーマに基づいて選択してください。"
            )
            
            lakeview_tips = dbx.run_chain(lakeview_prompt, genie_questions=st.session_state.genie_questions, general_schema=st.session_state.general_schema)
            st.session_state.lakeview_tips = lakeview_tips
                
        if "genie_questions" in st.session_state:
            st.markdown("### Genie サンプルプロンプト")
            
            # Display as list
            questions = st.session_state.genie_questions.strip().split("\n")
            for q in questions:
                if q:
                    st.write(f"- {q}")
        
        if "lakeview_tips" in st.session_state:
            st.markdown("### ダッシュボードのサンプルプロンプト")
            st.write(st.session_state.lakeview_tips)

# Model Training page
elif page == "モデル学習":
    st.header("機械学習モデルのトレーニング")
    
    if "synthetic_table" not in st.session_state:
        st.warning("先に合成データを生成してください。")
    elif "problem_type" not in st.session_state or "target_column" not in st.session_state:
        st.warning("問題タイプとターゲット列はデータ生成セクションで決定されるべきです。戻ってそのステップを完了してください。")
    else:
        # Display information about the model to be trained
        st.subheader("学習設定")
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.problem_type=="regress":
                st.info(f"**問題タイプ:** 回帰")
            elif st.session_state.problem_type=="classify":
                st.info(f"**問題タイプ:** 分類")
            else:
                st.info(f"**問題タイプ:** 未対応")
        
        with col2:
            st.info(f"**ターゲット列:** {st.session_state.target_column}")

        if st.button("モデル学習ジョブを作成"):
            with st.spinner("モデル学習ジョブを作成中..."):
                # Create job configuration
                suffix = randomname(5)
                job_name = f"{st.session_state.target_column}_{st.session_state.problem_type}_model_training"
                st.session_state.experiment_name = f"{st.session_state.problem_type}_{st.session_state.target_column}_{suffix}_experiment"
                st.session_state.model_name = f"{st.session_state.problem_type}_{st.session_state.target_column}"
                job_config = {
                    "name": job_name,
                    "tasks": [{
                        "task_key": "train_model",
                        "notebook_task": {
                            "notebook_path": dbx.root_dir+"/model_training",
                             "base_parameters": {
                                "catalog": dbx.catalog,
                                "schema": dbx.schema,
                                "user": dbx.user_name,
                                "table_name": st.session_state.synthetic_table,
                                "experiment_name": st.session_state.experiment_name,
                                "model_name": st.session_state.model_name,
                                "target_column": st.session_state.target_column,
                                "problem_type": st.session_state.problem_type
                            }
                        },
                        "existing_cluster_id": dbx.cluster_id
                    }]
                }
            
                # Create job
                job = dbx.create_job(job_config)
                job_id = job.job_id
                
                # Run job
                run = dbx.run_job(job_id)
                run_id = run.run_id

                # Get run information
                run_info = dbx.get_run(run_id)
                current_status = dbx.get_run_status(run_info)

                # Save information for later use
                st.session_state.model_job_id = job_id
                st.session_state.model_run_id = run_id

            with st.spinner("実験を取得中..."):
                # Get experiment ID from experiment name
                while True:
                    try:
                        experiment_name_fullpath = f"/demoarigato/databricks_automl/{st.session_state.experiment_name}"
                        experiment_response = dbx.get_experiment_by_name(experiment_name_fullpath)
                        if experiment_response is not None:
                            experiment_id = experiment_response.experiment.experiment_id
                            break
                    except:
                        time.sleep(10)
                        continue

                # Set experiment information
                st.session_state.experiment_id = experiment_id 
                st.session_state.experiment_url = f"{dbx.databricks_host}ml/experiments/{st.session_state.experiment_id}"
                st.session_state.model_url = f"{dbx.databricks_host}explore/data/models/{dbx.catalog}/{dbx.schema}/{st.session_state.model_name}/version/1"
                
                # Display experiment
                st.markdown("### 実験の進捗")
                st.markdown(f"- [Databricksで実験を表示]({st.session_state.experiment_url})")

            with st.spinner("モデルを学習中..."):
                # Wait for training completion
                while True:
                    time.sleep(20)
                    run_info = dbx.get_run(run_id)
                    current_status = dbx.get_run_status(run_info)
                    if current_status == "TERMINATED":
                        experiment_response = dbx.get_experiment_by_name(experiment_name_fullpath)
                        notebook_path = dbx.get_best_run_notebook(experiment_response)
                        break

                # Display notebook path
                st.markdown(f"- [Databricksでノートブックを表示]({notebook_path})")

                 # Get best run notebook path
                st.session_state.best_notebook_path = notebook_path
                
                # Display model URL
                st.success("学習が完了しました")
                st.markdown(f"- [Unity Catalogでモデルを表示]({st.session_state.model_url})")

                # Set completion flag
                st.session_state.model_trained = True
        else:
            if "experiment_id" in st.session_state:
                # Generate Databricks resource URLs - updated with actual IDs from job output
                st.markdown("### モデルリソース（ジョブ完了後に利用可能）")
                st.markdown(f"- [Databricksで実験を表示]({st.session_state.experiment_url})")
                st.markdown(f"- [Unity Catalogでモデルを表示]({st.session_state.model_url})")
                st.markdown(f"- [Databricksでノートブックを表示]({notebook_path})")
                
# Job Execution page
elif page == "ジョブ実行":
    st.header("エンドツーエンドパイプラインの作成と実行")
    
    if "model_trained" not in st.session_state or "source_schema" not in st.session_state:
        st.warning("先にモデルを学習し、パイプラインスキーマが定義されていることを確認してください。")
    else:
        if st.button("エンドツーエンドパイプラインジョブを作成"):
            with st.spinner("ソースデータを作成中..."):
                # Create job configuration
                job_config = {
                    "name": "メダリオンパイプライン",
                    "tasks": [{
                        "task_key": "medallion",
                        "notebook_task": {
                            "notebook_path": dbx.root_dir+"/medallion",
                             "base_parameters": {
                                "catalog": dbx.catalog,
                                "schema": dbx.schema,
                                "synthetic_table": st.session_state.synthetic_table,
                                "source_schema": st.session_state.source_schema,
                                "warehouse_id": dbx.warehouse_id,
                            }
                        },
                        "existing_cluster_id": dbx.cluster_id
                    }]
                }
            
                # Create job
                job = dbx.create_job(job_config)
                job_id = job.job_id
                
                # Run job
                run = dbx.run_job(job_id)
                run_id = run.run_id

                # Wait for job completion and get results
                retries = 0
                while True:
                    time.sleep(10)
                    run_info = dbx.get_run(run_id)
                    current_status = dbx.get_run_status(run_info)
                    if current_status == "TERMINATED":
                        try:
                            notebook_output = dbx.get_run_output(run_info)
                            result = json.loads(notebook_output)
                            break
                        except:
                            retries += 1
                            time.sleep(10)
                            if retries > 10:
                                raise Exception("ノートブック出力の取得に失敗しました")
                    elif current_status == "FAILED":
                        raise Exception("ジョブが失敗しました")
                            
            with st.spinner("エンドツーエンドパイプラインを作成中..."):
                # Get SQL queries
                sql_queries = {
                        "create_silver_1": result["silver_table_queries"][0],
                        "create_silver_2": result["silver_table_queries"][1],
                        "create_gold": result["gold_table_query"]
                    }
                
                # Update job configuration
                job_config = dbx.update_job_config(
                        sql_queries=sql_queries,
                        warehouse_id=dbx.warehouse_id,
                        notebook_path=st.session_state.best_notebook_path,
                        cluster_id=dbx.cluster_id
                    )
                
                # Create and run job
                job = dbx.create_job(job_config)
                job_id = job.job_id
                
                run = dbx.run_job(job_id)
                run_id = run.run_id
                
                # Save information
                st.session_state.job_created = True
                st.session_state.job_id = job_id
                st.session_state.job_run_id = run_id
                
                # Create URLs
                job_url = f"{dbx.databricks_host}jobs/{job_id}"
                run_url = f"{dbx.databricks_host}jobs/runs/show/{run_id}"
                st.session_state.job_url = job_url
                
                # Display results
                st.success(f"エンドツーエンドパイプラインジョブを作成して開始しました")
                st.markdown(f"- [Databricksでジョブを表示]({job_url})")
                st.markdown(f"- [Databricksで実行を表示]({run_url})")

# Demo Script page
elif page == "デモスクリプト":
    st.header("デモスクリプトの生成")
    
    if "job_created" not in st.session_state:
        st.warning("先にエンドツーエンドパイプラインジョブを作成してください。")
    else:
        if st.button("デモスクリプトを生成"):
            with st.spinner("デモスクリプトを生成中..."):
                # Define prompt template for demo script generation
                script_prompt = PromptTemplate(
                    input_variables=[
                        "business_question", "storyline", "general_schema", 
                        "genie_questions", "lakeview_tips", "target_column", 
                        "problem_type", "source_schema", "synthetic_table_url", 
                        "experiment_url", "model_url", "best_notebook_url", "job_url"
                    ],
                    template="""
                    # Databricksデモンストレーションスクリプト生成プロンプト

                    あなたはDatabricksのデモンストレーション専門家です。以下の要素を含む、Databricksの魅力と機能を効果的に伝えるデモンストレーションスクリプトを作成してください。

                    ## 1. ビジネス課題と背景
                    - ビジネス課題: {business_question}
                    - 背景ストーリー: {storyline}

                    ## 2. データ分析基盤
                    - 分析用データ構造: {general_schema}
                    - Unity Catalogの活用法:
                      * データ権限管理
                      * カタログ機能
                      * 参照URL: {synthetic_table_url}

                    ## 3. AI/BI Genie機能
                    - Genie Data Roomの作成と活用デモ
                    - サンプル質問例: {genie_questions}

                    ## 4. AI/BIダッシュボード
                    - ダッシュボード作成手順
                    - 作成プロンプト: {lakeview_tips}

                    ## 5. 機械学習機能
                    - 問題タイプ: {problem_type}
                    - 予測対象: {target_column}
                    - AutoMLの活用方法
                    - MLflow Experimentsの活用: {experiment_url}
                    - Unity Catalogでのモデル管理: {model_url}

                    ## 6. Databricks Workflows
                    - ジョブ管理の概要
                    - パイプライン構成: 
                      * シルバーテーブルの定義 ({source_schema})
                      * ゴールドテーブルの定義 ({general_schema})
                      * AutoMLで生成したモデル訓練ノートブック({best_notebook_url})
                    - 最終実行ノートブック: {job_url}

                    デモンストレーションは各セクションを論理的に接続し、ビジネス課題がDatabricksによってどのように解決されるかを明確に示してください。
                    """
                )
                
                # Generate demo script
                demo_script = dbx.run_chain(
                    script_prompt,
                    max_tokens=10000,
                    business_question=st.session_state.get("business_question", ""),
                    storyline=st.session_state.get("storyline", ""),
                    general_schema=st.session_state.get("general_schema", ""),
                    genie_questions=st.session_state.get("genie_questions", ""),
                    lakeview_tips=st.session_state.get("lakeview_tips", ""),
                    target_column=st.session_state.get("target_column", ""),
                    problem_type=st.session_state.get("problem_type", ""),
                    source_schema=st.session_state.get("source_schema", ""),
                    synthetic_table_url=st.session_state.get("synthetic_table_url", ""),
                    experiment_url=st.session_state.get("experiment_url", ""),
                    model_url=st.session_state.get("model_url", ""),
                    best_notebook_url=st.session_state.get("best_notebook_url", ""),
                    job_url=st.session_state.get("job_url", "")
                )
                
                st.session_state.demo_script = demo_script
        
        if "demo_script" in st.session_state:
            st.subheader("生成されたデモスクリプト")
            st.markdown(st.session_state.demo_script)
            
            if st.download_button(
                label="スクリプトをダウンロード",
                data=st.session_state.demo_script,
                file_name="databricks_demo_script.md",
                mime="text/markdown"
            ):
                st.success("スクリプトが正常にダウンロードされました！")