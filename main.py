import argparse 
import ollama
from loguru import logger
import pandas as pd
import numpy as np
import plotly.express as px
import os
import json
import io
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path


@dataclass
class EDAResults:
    """Data class to store EDA results"""
    shape: Tuple[int,int]
    columns: List[str]
    columns_with_null_values: Dict[str,int]
    info_about_data: str
    store_corr_matrix:List[Tuple[str,str,float]]
    dataframe: pd.DataFrame
    datatype_issues: Dict[str, List[str]]
    categorical_analysis: Dict[str, Dict[str, Any]]


class DataAnalyzer:
    """Data cleaning and preprocessing"""
    def __init__(self, max_unique_threshold: int = 20):
        self.max_unique_threshold = max_unique_threshold

    def analyze_correlations(self, df:pd.DataFrame) -> List[Tuple[str,str,float]]:

        """find correlations between columns for analyze_df"""
        # data_correlation = df.corr().to_numpy()
        numeric_df = df.select_dtypes(include=np.number)
        store_corr_columns = []
        # for i in range(len(data_correlation)):
        #     for j in range(len(data_correlation[0])):
        #         if abs(data_correlation[i][j]) >= 0.5:
        #             store_corr_columns.append((column_names[i], column_names[j], data_correlation[i][j]))
        if not numeric_df.empty:
            corr_matrix = numeric_df.corr().abs()  # Absolute correlations
            high_corr = corr_matrix[corr_matrix > 0.5].stack().reset_index()
            high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
            high_corr = high_corr.drop_duplicates()
            
            for _, row in high_corr.iterrows():
                col1, col2, corr_value = row['level_0'], row['level_1'], row[0]
                store_corr_columns.append((col1, col2, float(corr_value)))
        
        # if store_corr_columns:
        #     plot_data(store_corr_columns, df)

        return store_corr_columns
    
    def check_datatypes(self, df:pd.DataFrame)->Dict[str, List[str]]:
        """Checks datatype of columns to detect date-time or numeric type stored in incorrect format"""
        print('16\n')
        issues = {}

        for col in df.columns:
            col_issues = []

            if df[col].dtype=='object':
                temp_vals = df[col].dropna().head(10).astype(str)
                if any(len(val)>8 and ('-' in val or '/' in val) for val in temp_vals):
                    col_issues.append("Potential datatime column stored as object")
                
                else:
                    try:
                        pd.to_numeric(df[col], errors='raise')
                        col_issues.append("Numeric data stored as object")
                    except:
                        temp_types = df[col].dropna().apply(type).unique()
                        if len(temp_types)>1:
                            col_issues.append(f"Mixed datatypes in {temp_types}")
            
            if col_issues:
                issues[col]=col_issues

        print('17\n')
        
        return issues

    def analyze_categorical_columns(self, df:pd.DataFrame)->Dict[str, Dict[str,Any]]:
        """Heuristic-based analysis of categorical-like columns"""
        cat_analysis = {}

        for col in df.columns:
            unique_vals = df[col].nunique(dropna=False)
            total_vals = len(df[col])
            dtype = df[col].dtype

            # Heuristic: consider any column with few unique values as categorical
            if dtype in ['object', 'category'] or unique_vals <= self.max_unique_threshold:
                cat_analysis[col] = {
                    'dtype': str(dtype),
                    'unique_count': unique_vals,
                    'unique_percentage': (unique_vals / total_vals) * 100,
                    'top_values': df[col].value_counts(dropna=False).head(5).to_dict(),
                    'is_high_cardinality': unique_vals > total_vals * 0.5,
                    'potential_id_column': unique_vals == total_vals,
                    'is_binary': unique_vals == 2
                }

        return cat_analysis

    
    def analyze_df(self, df: pd.DataFrame)->EDAResults:
        """Performs data analysis"""
        # df = pd.read_csv(filetype)
        no_of_records, no_of_columns = df.shape

        column_names = df.columns.to_list()
        check_null_value_columns = df.isnull().sum()
        check_null_value_columns = check_null_value_columns[check_null_value_columns > 0].to_dict()

        # data_info = df.info().to_string()
        buffer = io.StringIO()
        df.info(buf=buffer)
        data_info = buffer.getvalue()

        # CHECK CORRELATIONS
        store_corr_columns = self.analyze_correlations(df)

        # CHECK DATATYPE MISMATCHES
        datatype_issues = self.check_datatypes(df)

        # CHECK CATEGORICAL COLUMNS
        categorical_data_analysis = self.analyze_categorical_columns(df)

        # return (no_of_records, no_of_columns), column_names, check_null_value_columns, data_info, store_corr_columns
        
        # return {
        #     "shape": (no_of_records, no_of_columns),
        #     "columns": column_names,
        #     "columns_with_null_values":check_null_value_columns,
        #     "info_about_data": data_info,
        #     "store_corr_matrix": store_corr_columns,
        #     "dataframe":df,
        #     "potential_datatype_issues":datatype_issues,
        #     "categorical_data":categorical_data_analysis
        # }

        return EDAResults(
            shape= (no_of_records, no_of_columns),
            columns= column_names,
            columns_with_null_values=check_null_value_columns,
            info_about_data= data_info,
            store_corr_matrix= store_corr_columns,
            dataframe=df,
            datatype_issues=datatype_issues,
            categorical_analysis=categorical_data_analysis
        )
    
class DataVisualizer:
    """Performs elementary data visualisation"""

    @staticmethod
    def create_scatter_plot(correlations: List[Tuple[str,str,float]], df:pd.DataFrame):
        """creates scatter plots for correlated variables"""
        if len(correlations)>=2:
            for x_col, y_col, corr_eval in correlations:
                try:
                    fig= px.scatter(
                        df, x_col, y_col,
                        title=f"Scatter Plot: {x_col} vs {y_col} (r={corr_eval:.2f})"
                    )
                    fig.show()
                
                except Exception as e:
                    logger.warning(f"Could not create scatter plot for {x_col} vs {y_col}: {e}")

    @staticmethod
    def create_distribution_plots(df: pd.DataFrame):
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # Histogram
                fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                fig_hist.show()
                
                # Box plot
                fig_box = px.box(df, y=col, title=f'Box Plot of {col}')
                fig_box.show()
            except Exception as e:
                logger.warning(f"Could not create distribution plots for {col}: {e}")

    @staticmethod
    def create_correlation_heatmap(df:pd.DataFrame):
        """Create an interactive correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            try:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix, 
                            title="Correlation Heatmap",
                            color_continuous_scale='RdBu_r',
                            aspect="auto")
                fig.show()
            except Exception as e:
                logger.warning(f"Could not create correlation heatmap: {e}")

    @staticmethod
    def plot_categorical_distributions(df: pd.DataFrame):
        """Plot distributions for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_cols:
            if df[col].nunique() <= 20:  # Only plot if not too many categories
                try:
                    value_counts = df[col].value_counts()
                    fig = px.bar(x=value_counts.index, y=value_counts.values,
                                title=f'Distribution of {col}')
                    fig.show()
                
                except Exception as e:
                    logger.warning(f"Could not create categorical plot for {col}: {e}")


class LLMAnalyzer:
    """Ollama based analysis"""

    def __init__(self, model_name: str="llama3.2:latest"):
        self.model_name = model_name

    def create_system_message(self, results: EDAResults)->str:
        """Creates the system prompt to be used by generate_recommendations"""
        return (
             f"You are a data analyst who has performed EDA on a csv file and obtained the following results:\n"
                    f"Shape of data: {results.shape}\n"
                    f"List of Columns in dataset: {results.columns}\n"
                    f"Columns with null values: {results.columns_with_null_values}\n"
                    f"Data Information: {results.info_about_data}\n"
                    f"High correlations found: {len(results.store_corr_matrix)} pairs\n"
                    f"Data type issues: {len(results.datatype_issues)} columns\n\n"
                    "Based on this data, present future steps (as pointers in plaintext) for proceeding with data cleaning and visualisation. "
                    "Select from:\n 1. Remove null values \n 2. Suggest visualisations between columns \n 3. Visualize correlation matrix, etc."
        )

    
    def generate_recommendations(self, results: EDAResults)->str:
        """Generate potential steps for doing EDA via LLM"""
        # system_message = self.create_system_message(results)
        system_message = self.create_system_message(results)

        messages = [
                    {
                        "role":"user",
                        "content": system_message,
                    }
                ]
        
        try:
            logger.info("Generating recommendations...\n")

            response = ollama.chat(
                model='llama3.2:latest',
                messages=messages,
                options={'num_ctx': 4096},  # Increase context window
                stream=False  # Disable streaming for simplicity
            )

            if response and 'message' in response and 'content' in response['message']:
                content = response['message']['content']
                print("LLM response received successfully")
                return content
            else:
                print("Empty response from LLM")
                return "No response from LLM"
            
        except Exception as e:
            logger.error(f"Error querying Ollama: {e}")
            return "Could not get advice from LLM."
    
    def test_ollama_connection(self):
        """Function to check whether Ollama is confugured on your system"""
        print("Testing Ollama connection...")
        try:
            response = ollama.chat(
                model='llama3.2:latest',
                messages=[{'role': 'user', 'content': 'Say "Hello World"'}]
            )
            if response and 'message' in response:
                print(f"Test successful! Response: {response['message']['content']}")
                return True
            else:
                print("Empty test response")
                return False
        except Exception as e:
            print(f"Test failed: {e}")
            return False
        

class ReportGenerator:
    """Handle report generation and saving"""

    @staticmethod
    def save_report(results: EDAResults, format_type: str="txt",output_path: str="eda_report"):
        """saves eda report to a file"""
        try:
            output_file = Path(f"{output_path}.{format_type}")


            if format_type=="json":
                ReportGenerator.save_json_report(results, output_file)

            else:
                ReportGenerator.save_text_report(results, output_file)
            
            logger.success(f"Report saved to {output_file}")

        except Exception as e:
            logger.error(f"Error saving file: {e}")

    
    @staticmethod
    def save_json_report(results:EDAResults, output_file: Path):
        """Save report in json format"""

        report_data = {
            "shape":results.shape,
            "columns":results.columns,
            "columns_with_null_values": results.columns_with_null_values,
            "info_about_data":results.info_about_data,
            "correlation_matrix": results.store_corr_matrix,
            "datatype_issues": results.datatype_issues,
            "categorical_analysis": results.categorical_analysis
        }

        with open(output_file, 'w') as f:
            json.dump(report_data, f, indent=4, default=str)


    @staticmethod
    def save_text_report(results:EDAResults, output_file: Path):
        """Save report in text format"""
        with open(output_file, 'w') as f:
            f.write("=== AUTO EDA REPORT ===\n\n")
            f.write(f"Dataset shape: {results.shape}\n\n")
            f.write(f"Columns: {results.columns}\n\n")
            f.write(f"Columns with null values: {results.columns_with_null_values}\n\n")
            f.write("Data Information:\n")
            f.write(f"{results.info_about_data}\n\n")
            f.write("High correlations (>0.5):\n")

            for x_col, y_col, corr in results.store_corr_matrix:
                f.write(f"{x_col}<->{y_col}: {corr:.2f}\n")

            f.write(f"Datatype Mismatches: {results.datatype_issues}\n")
            f.write(f"\nCategorical Analysis: {results.categorical_analysis}\n")


class DataLoader:
    """Loads all types of Data formats
    Currently supports CSVs"""

    @staticmethod
    def load_data(file_path:str, file_type:str)->pd.DataFrame:
        """Loads data from file"""
        if file_type.lower()=="csv":
            return pd.read_csv(file_path)

        else:
            raise NotImplementedError(f"Support for {file_type} files is coming soon!")


class AutoEDA:
    """Main class for performing EDA"""

    def __init__(self, model_name:str="llama3.2:latest"):
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer
        self.llm_analyzer = LLMAnalyzer()
        self.report_generator = ReportGenerator()
        self.data_loader = DataLoader()

    def run_eda(self, file_path: str, file_type:str, use_llm: bool=False, visualize: bool=False, save_report: bool=True, output_format: str="txt")->EDAResults:
        """Orchestrates the EDA process"""

        try:
            # LOAD DATA
            logger.info(f"Loading {file_type} file: {file_path}")

            df = self.data_loader.load_data(file_path, file_type)

            # ANALYZE DATA
            results = self.analyzer.analyze_df(df)

            # GENERATE VISUALIZATIONS
            if visualize:
                logger.info("Creating visualizations...")
                self.visualizer.create_scatter_plots(results.store_corr_matrix, df)
                self.visualizer.create_distribution_plots(df)
                self.visualizer.create_correlation_heatmap(df)
                self.visualizer.create_categorical_plots(df)

            # GENERATE LLM RECOMMENDATIONS IF REQUESTED
            if use_llm: 
                if self.llm_analyzer.test_ollama_connection():
                    recommendations = self.llm_analyzer.generate_recommendations(results)
                    print("\n=== Future Steps===\n")
                    print(recommendations)
                
                else:
                    logger.error("LLM connection failed. Skipping recommendations")

            # Save report if requested
            if save_report:
                self.report_generator.save_report(results, output_format)

            return results

        except Exception as e:
            logger.error(f"Error during EDA process: {e}")
            logger.error(traceback.format_exc())
            raise


    def print_basic_results(self, results:EDAResults):
        """Prints basic results to console"""

        print("\n====EDA RESULTS====\n")
        print(f"Shape: {results.shape}\n")
        print(f"Columns: {results.columns}\n")
        print(f"Columns with Nulls: {results.columns_with_null_values}\n")
        print(f"High correlations: {len(results.store_corr_matrix)} pairs found:\n")

        for x_col, y_col, corr in results.store_corr_matrix:
            print(f"{x_col}<->{y_col}: {corr:.2f}")

        if results.datatype_issues:
            print(f"\n Potential type mismatches found in {len(results.datatype_issues)} columns:")
            for col, issues in results.datatype_issues.items():
                print(f"{col}: {','.join(issues)}")

def main():
    """Main function for CLI"""
    
    parser = argparse.ArgumentParser(
        description="Perform automatic EDA for your CSV files",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
    parser.add_argument("--file_type", type=str, required=True, help="File extension type(eg 'csv')")
    parser.add_argument("--use_llm", action="store_true", help="Use local LM to provide EDA suggestions")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations for the data")
    parser.add_argument("--save_report", action="store_true", help="Save the EDA output to file")
    parser.add_argument("--output_format", choices=["txt","json"], default="txt", help="Output report format")
    parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model to be used for analysis")

    args=parser.parse_args()

    try:
        if not args.file_path or not args.file_type:
            raise ValueError("You either forgot to specify the file type or the file path.")

        # INITIALISE AND RUN AUTO-EDA
        auto_eda = AutoEDA(model_name = args.model)

        auto_eda.run_eda(
            file_path = args.file_path,
            file_type = args.file_type,
            use_llm = args.use_llm,
            visualize = args.visualize,
            save_report = args.save_report,
            output_format = args.output_format
        )

    except Exception as e:
        logger.error("You did something wrong somewhere my bro:{e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()



# def plot_data(store_corr_columns, df):
#     print("1\n")
#     if len(store_corr_columns)>=2:
#         for i in range(len(store_corr_columns)):
#             col_x, col_y, _ = store_corr_columns[i]
#             try:
#                 fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot of {col_x} vs {col_y}")
#                 fig.show()
#             except Exception as e:
#                 logger.warning(f"Damn bruh we can't plot your scatter plot cuz *drum roll please*: {e}")
    
#     else:
#         # plot pie chart for all labels in label type columns if --visualise set to true
#         pass



# def create_correlation_heatmap(df):
#     """Create an interactive correlation heatmap"""
#     numeric_df = df.select_dtypes(include=[np.number])
#     if not numeric_df.empty:
#         corr_matrix = numeric_df.corr()
#         fig = px.imshow(corr_matrix, 
#                        title="Correlation Heatmap",
#                        color_continuous_scale='RdBu_r',
#                        aspect="auto")
#         fig.show()

# def plot_categorical_distributions(df):
#     """Plot distributions for categorical columns"""
#     categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
#     for col in categorical_cols:
#         if df[col].nunique() <= 20:  # Only plot if not too many categories
#             value_counts = df[col].value_counts()
#             fig = px.bar(x=value_counts.index, y=value_counts.values,
#                         title=f'Distribution of {col}')
#             fig.show()

# def analyze_categorical_columns(df, max_unique_threshold=20):
#     """Heuristic-based analysis of categorical-like columns"""
#     cat_analysis = {}

#     for col in df.columns:
#         unique_vals = df[col].nunique(dropna=False)
#         total_vals = len(df[col])
#         dtype = df[col].dtype

#         # Heuristic: consider any column with few unique values as categorical
#         if dtype in ['object', 'category'] or unique_vals <= max_unique_threshold:
#             cat_analysis[col] = {
#                 'dtype': str(dtype),
#                 'unique_count': unique_vals,
#                 'unique_percentage': (unique_vals / total_vals) * 100,
#                 'top_values': df[col].value_counts(dropna=False).head(5).to_dict(),
#                 'is_high_cardinality': unique_vals > total_vals * 0.5,
#                 'potential_id_column': unique_vals == total_vals,
#                 'is_binary': unique_vals == 2
#             }

#     return cat_analysis


# def check_datatype(df):
#     """Checks datatype of columns to detect date-time or numeric type stored in incorrect format"""
#     print('16\n')
#     issues = {}

#     for col in df.columns:
#         col_issues = []

#         if df[col].dtype=='object':
#             temp_vals = df[col].dropna().head(10).astype(str)
#             if any(len(val)>8 and ('-' in val or '/' in val) for val in temp_vals):
#                 col_issues.append("Potential datatime column stored as object")
            
#             else:
#                 try:
#                     pd.to_numeric(df[col], errors='raise')
#                     col_issues.append("Numeric data stored as object")
#                 except:
#                     temp_types = df[col].dropna().apply(type).unique()
#                     if len(temp_types)>1:
#                         col_issues.append(f"Mixed datatypes in {temp_types}")
        
#         if col_issues:
#             issues[col]=col_issues

#     print('17\n')
    
#     return issues


# def call_summarise_csv(filetype):
#     print("2\n")
#     df = pd.read_csv(filetype)
#     no_of_records, no_of_columns = df.shape

#     column_names = df.columns.to_list()
#     check_null_value_columns = df.isnull().sum()
#     check_null_value_columns = check_null_value_columns[check_null_value_columns > 0].to_dict()

#     # data_info = df.info().to_string()
#     buffer = io.StringIO()
#     df.info(buf=buffer)
#     data_info = buffer.getvalue()

#     # data_correlation = df.corr().to_numpy()
#     numeric_df = df.select_dtypes(include=np.number)
#     store_corr_columns = []
#     # for i in range(len(data_correlation)):
#     #     for j in range(len(data_correlation[0])):
#     #         if abs(data_correlation[i][j]) >= 0.5:
#     #             store_corr_columns.append((column_names[i], column_names[j], data_correlation[i][j]))
#     if not numeric_df.empty:
#         corr_matrix = numeric_df.corr().abs()  # Absolute correlations
#         high_corr = corr_matrix[corr_matrix > 0.5].stack().reset_index()
#         high_corr = high_corr[high_corr['level_0'] != high_corr['level_1']]
#         high_corr = high_corr.drop_duplicates()
        
#         for _, row in high_corr.iterrows():
#             col1, col2, corr_value = row['level_0'], row['level_1'], row[0]
#             store_corr_columns.append((col1, col2, float(corr_value)))
#     # if store_corr_columns:
#     #     plot_data(store_corr_columns, df)

#     datatype_issues = check_datatype(df)
#     categorical_data_analysis = analyze_categorical_columns(df)

#     # return (no_of_records, no_of_columns), column_names, check_null_value_columns, data_info, store_corr_columns
#     return {
#         "shape": (no_of_records, no_of_columns),
#         "columns": column_names,
#         "columns_with_null_values":check_null_value_columns,
#         "info_about_data": data_info,
#         "store_corr_matrix": store_corr_columns,
#         "dataframe":df,
#         "potential_datatype_issues":datatype_issues,
#         "categorical_data":categorical_data_analysis
#     }


# def analyse_file(filepath, filetype):
#     print("3\n")
#     if filetype == "csv":
#         return call_summarise_csv(filepath)
#     else:
#         print(f"We're bringing in support for {filetype} files. Hang in there bud!")
#         return
    
# def generate_llm_advice(summary):
#     print("4\n")
#     system_message = (f"You are a data analyst who has just performed EDA on a csv file and obtained the following results:\n"
#                     f"Shape of data: {summary['shape']}\n"
#                     f"List of Columns in dataset: {summary['columns']}\n"
#                     f"Columns with null values: {summary['columns_with_null_values']}\n"
#                     f"Data Information: {summary['info_about_data']}\n"
#                     f"Correlation Matrix: {summary['store_corr_matrix']}\n"
#                     "Based on this data, present future steps (as pointers in plaintext) for proceeding with data cleaning and visualisation. "
#                     "Select from:\n 1. Remove null values \n 2. Suggest visualisations between columns \n 3. Visualize correlation matrix, etc."
#                     )
#     # system_message = (
#     #                 "You are a data analyst reviewing EDA results. "
#     #                 f"Dataset has {summary['shape'][0]} rows and {summary['shape'][1]} columns.\n"
#     #                 f"Columns: {', '.join(summary['columns'])}\n"
#     #                 f"Null values found in: {summary['columns_with_null_values'] or 'None'}\n"
#     #                 "Suggest 3-5 specific next steps for data cleaning and visualization. "
#     #                 "Be concise and focus on actionable insights."
#     #                 )
#     print("11\n")
    
#     messages = [
#                     {
#                         "role":"user",
#                         "content": system_message,
#                     }
#                 ]
#     print("12\n")
#     try:
#         print("Sending to LLM...")
#         response = ollama.chat(
#             model='llama3.2:latest',
#             messages=messages,
#             options={'num_ctx': 4096},  # Increase context window
#             stream=False  # Disable streaming for simplicity
#         )

#         if response and 'message' in response and 'content' in response['message']:
#             content = response['message']['content']
#             print("LLM response received successfully")
#             return content
#         else:
#             print("Empty response from LLM")
#             return "No response from LLM"



#     except Exception as e:
#         logger.error(f"Error querying Ollama: {e}")
#         return "Could not get advice from LLM."
    
# def save_report(summary, format="txt", output_path="eda_report"):
#     print("5\n")
#     try:
#         if format=="json":
#             with open(f"{output_path}.json","w") as f:
#                 json.dump({k: v for k, v in summary.items() if k!="dataframe"},f,indent=4)
        
#         else:
#             with open(f"{output_path}.txt","w") as f:
#                 f.write("Auto EDA report\n")
#                 f.write(f"Shape: {summary['shape']}\n")
#                 f.write(f"Columns: {summary['columns']}\n")
#                 f.write(f"Columns with null values: {summary['columns_with_null_values']}\n")
#                 f.write(f"Data stats: {summary['info_about_data']}\n")
#                 f.write("Correlation Matrix:\n")
#                 for row in summary['store_corr_matrix']:
#                     f.write(f"{row}\n")

#         logger.success(f"Report saved to {output_path}.{format}")

#     except Exception as e:
#         logger.error(f"Error saving my bro: {e}")
        

# def test_ollama_connection():
#     print("Testing Ollama connection...")
#     try:
#         response = ollama.chat(
#             model='llama3.2:latest',
#             messages=[{'role': 'user', 'content': 'Say "Hello World"'}]
#         )
#         if response and 'message' in response:
#             print(f"Test successful! Response: {response['message']['content']}")
#             return True
#         else:
#             print("Empty test response")
#             return False
#     except Exception as e:
#         print(f"Test failed: {e}")
#         return False


# def main():
#     print("6\n")
#     # Test Ollama connection first

#     # if not test_ollama_connection():
#     #     logger.error("Ollama connection test failed. Cannot proceed with LLM analysis.")
#     #     return
#     parser = argparse.ArgumentParser(description="Perform automatic EDA for your CSV files")

#     parser.add_argument("--file_path", type=str, required=True, help="Path of the csv file")
#     parser.add_argument("--file_type", type=str, required=True, help="file extension type")
#     parser.add_argument("--use_llm", action="store_true", help="Use ollama to provide EDA suggestions")
#     parser.add_argument("--visualise", action="store_true", help="Assisted visualisations for data")
#     parser.add_argument("--save_report", action="store_true", help="Save the EDA output to file")
#     parser.add_argument("--output_format", choices=["txt", "json"], default="txt", help="Output report format")


#     args = parser.parse_args()

#     try:
#         if not args.file_path or not args.file_type:
#             raise ValueError("You either forgot to specify the file type or the file path.")
        
#         if args.use_llm is True:
#                 # data_shape, column_list, columns_with_null_values, info_about_data, store_corr_matrix = analyse_file(args.file_path, args.file_type)
#             summary = analyse_file(args.file_path, args.file_type)

#             print(f"\n Shape: {summary['shape']}")
#             print(f"\nColumns: {summary['columns']}")
#             print(f"\nColumns with nulls: {summary['columns_with_null_values']}")
#             print(f"\nCorrelated Columns (> 0.5 or < -0.5): {summary['store_corr_matrix']}\n")
#             for corr in summary['store_corr_matrix']:
#                 print(f"  {corr[0]} vs {corr[1]}: {corr[2]:.2f}")

#             if args.visualise:
#                 plot_data(summary['store_corr_matrix'], summary['dataframe'])

#             if args.use_llm:
#                 response = generate_llm_advice(summary)
#                 print("\n Aight buckle up buddy this is what llama says:\n")
#                 print("15\n")
#                 print(response)

#             if args.save_report:
#                 save_report(summary, format=args.output_format)
                
        
#     except Exception as e:
#         logger.error("You did something wrong somewhere my bro:{e}")
#         logger.error(traceback.format_exc())

# if __name__ == "__main__":
#     main()