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

def plot_data(store_corr_columns, df):
    print("1\n")
    if len(store_corr_columns)>=2:
        for i in range(len(store_corr_columns)):
            col_x, col_y, _ = store_corr_columns[i]
            try:
                fig = px.scatter(df, x=col_x, y=col_y, title=f"Scatter Plot of {col_x} vs {col_y}")
                fig.show()
            except Exception as e:
                logger.warning(f"Damn bruh we can't plot your scatter plot cuz *drum roll please*: {e}")
    
    else:
        # plot pie chart for all labels in label type columns if --visualise set to true
        pass


def call_summarise_csv(filetype):
    print("2\n")
    df = pd.read_csv(filetype)
    no_of_records, no_of_columns = df.shape

    column_names = df.columns.to_list()
    check_null_value_columns = df.isnull().sum()
    check_null_value_columns = check_null_value_columns[check_null_value_columns > 0].to_dict()

    # data_info = df.info().to_string()
    buffer = io.StringIO()
    df.info(buf=buffer)
    data_info = buffer.getvalue()

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

    # return (no_of_records, no_of_columns), column_names, check_null_value_columns, data_info, store_corr_columns
    return {
        "shape": (no_of_records, no_of_columns),
        "columns": column_names,
        "columns_with_null_values":check_null_value_columns,
        "info_about_data": data_info,
        "store_corr_matrix": store_corr_columns,
        "dataframe":df
    }


def analyse_file(filepath, filetype):
    print("3\n")
    if filetype == "csv":
        return call_summarise_csv(filepath)
    else:
        print(f"We're bringing in support for {filetype} files. Hang in there bud!")
        return
    
def generate_llm_advice(summary):
    print("4\n")
    # system_message = (f"You are a data analyst who has just performed EDA on a csv file and obtained the following results:\n"
    #                 f"Shape of data: {summary['shape']}\n"
    #                 f"List of Columns in dataset: {summary['columns']}\n"
    #                 f"Columns with null values: {summary['columns_with_null_values']}\n"
    #                 f"Data Information: {summary['info_about_data']}\n"
    #                 f"Correlation Matrix: {summary['store_corr_matrix']}\n"
    #                 "Based on this data, present future steps for proceeding with data cleaning and visualisation. "
    #                 "Select from:\n 1. Remove null values \n 2. Suggest visualisations between columns \n 3. Visualize correlation matrix, etc."
    #                 )
    system_message = (
                    "You are a data analyst reviewing EDA results. "
                    f"Dataset has {summary['shape'][0]} rows and {summary['shape'][1]} columns.\n"
                    f"Columns: {', '.join(summary['columns'])}\n"
                    f"Null values found in: {summary['columns_with_null_values'] or 'None'}\n"
                    "Suggest 3-5 specific next steps for data cleaning and visualization. "
                    "Be concise and focus on actionable insights."
                    )
    print("11\n")
    
    messages = [
                    {
                        "role":"user",
                        "content": system_message,
                    }
                ]
    print("12\n")
    try:
        print("Sending to LLM...")
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
    
def save_report(summary, format="txt", output_path="eda_report"):
    print("5\n")
    try:
        if format=="json":
            with open(f"{output_path}.json","w") as f:
                json.dump({k: v for k, v in summary.items() if k!="dataframe"},f,indent=4)
        
        else:
            with open(f"{output_path}.txt","w") as f:
                f.write("Auto EDA report\n")
                f.write(f"Shape: {summary['shape']}\n")
                f.write(f"Columns: {summary['columns']}\n")
                f.write(f"Columns with null values: {summary['columns_with_null_values']}\n")
                f.write(f"Data stats: {summary['info_about_data']}\n")
                f.write("Correlation Matrix:\n")
                for row in summary['store_corr_matrix']:
                    f.write(f"{row}\n")

        logger.success(f"Report saved to {output_path}.{format}")

    except Exception as e:
        logger.error(f"Error saving my bro: {e}")

def test_ollama_connection():
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


def main():
    print("6\n")
    # Test Ollama connection first

    if not test_ollama_connection():
        logger.error("Ollama connection test failed. Cannot proceed with LLM analysis.")
        return
    parser = argparse.ArgumentParser(description="Perform automatic EDA for your CSV files")

    parser.add_argument("--file_path", type=str, required=True, help="Path of the csv file")
    parser.add_argument("--file_type", type=str, required=True, help="file extension type")
    parser.add_argument("--use_llm", action="store_true", help="Use ollama to provide EDA suggestions")
    parser.add_argument("--visualise", action="store_true", help="Assisted visualisations for data")
    parser.add_argument("--save_report", action="store_true", help="Save the EDA output to file")
    parser.add_argument("--output_format", choices=["txt", "json"], default="txt", help="Output report format")


    args = parser.parse_args()

    try:
        if not args.file_path or not args.file_type:
            raise ValueError("You either forgot to specify the file type or the file path.")
        
        if args.use_llm is True:
                # data_shape, column_list, columns_with_null_values, info_about_data, store_corr_matrix = analyse_file(args.file_path, args.file_type)
                summary = analyse_file(args.file_path, args.file_type)

                print(f"\n Shape: {summary['shape']}")
                print(f"\nColumns: {summary['columns']}")
                print(f"\nColumns with nulls: {summary['columns_with_null_values']}")
                print(f"\nCorrelated Columns (> 0.5 or < -0.5): {summary['store_corr_matrix']}\n")
                for corr in summary['store_corr_matrix']:
                    print(f"  {corr[0]} vs {corr[1]}: {corr[2]:.2f}")

                if args.visualise:
                    plot_data(summary['store_corr_matrix'], summary['dataframe'])

                if args.use_llm:
                    response = generate_llm_advice(summary)
                    print("\n Aight buckle up buddy this is what llama says:\n")
                    print("15\n")
                    print(response)

                if args.save_report:
                    save_report(summary, format=args.output_format)
                
        
    except Exception as e:
        logger.error("You did something wrong somewhere my bro:{e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()