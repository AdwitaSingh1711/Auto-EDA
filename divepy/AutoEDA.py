import argparse 
import ollama
from loguru import logger
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import io
import traceback
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from plotly.subplots import make_subplots
import webbrowser
import platform
import subprocess

from .DataAnalyzer import DataAnalyzer
from .DataLoader import DataLoader
from .DataVisualizer import DataVisualizer
from .EDAResults import EDAResults
from .LLMAnalyzer import LLMAnalyzer
from .ReportGenerator import ReportGenerator


class AutoEDA:
    """Main class for performing EDA"""

    def __init__(self, model_name:str="llama3.2:latest", plot_output_dir: str = "eda_plots", output_path: str = "eda_report"):
        self.analyzer = DataAnalyzer()
        self.visualizer = DataVisualizer(output_dir = plot_output_dir)
        self.llm_analyzer = LLMAnalyzer()
        self.report_generator = ReportGenerator(output_path = output_path)
        self.data_loader = DataLoader()

    def run_eda(self, file_path: str, file_type:str, use_llm: bool=False, visualize: bool=False, save_report: bool=True, output_format: str="txt", output_path: str="eda_report", plot_output_dir: str="eda_plots")->EDAResults:
        """Orchestrates the EDA process"""

        try:
            # LOAD DATA
            logger.info(f"Loading {file_type} file: {file_path}")

            df = self.data_loader.load_data(file_path, file_type)

            # ANALYZE DATA
            results, cleaned_data = self.analyzer.analyze_df(df)

            # SAVES PREPROCESSED CSV TO OUTPUT_PATH
            df.to_csv(f"{output_path}/cleaned_data.csv", index = False)

            # GENERATE VISUALIZATIONS
            if visualize:
                logger.info("Creating visualizations...")
                # self.visualizer.create_scatter_plot(results.store_corr_matrix, df)
                self.visualizer.create_scatter_plot(results.store_corr_matrix, cleaned_data)
                # self.visualizer.create_distribution_plots(df)
                self.visualizer.create_distribution_plots(cleaned_data)
                # self.visualizer.create_correlation_heatmap(df)
                self.visualizer.create_correlation_heatmap(cleaned_data)


                # self.visualizer.create_categorical_plots(df)
                # Linking Cat_analysis 
                # self.visualizer.plot_categorical_distributions(df, results.categorical_analysis)
                self.visualizer.plot_categorical_distributions(cleaned_data, results.categorical_analysis)

                index_path = self.visualizer.generate_plot_index()
                self.visualizer.open_report_in_browser(index_path)
                self.visualizer.print_visualization_summary()

            # GENERATE LLM RECOMMENDATIONS IF REQUESTED
            recommendations = ""
            if use_llm: 
                if self.llm_analyzer.test_ollama_connection():
                    recommendations = self.llm_analyzer.generate_recommendations(results)
                    # print("\n=== Future Steps===\n")
                    # print(recommendations)
                
                else:
                    logger.error("LLM connection failed. Skipping recommendations")

            # Save report if requested
            if save_report:
                # self.report_generator.save_report(results, output_format, output_path)
                if recommendations:
                    # self.report_generator.save_report(results, output_format, output_path, recommendations)
                    self.report_generator.save_report(results, output_format, recommendations)
                else:
                    # self.report_generator.save_report(results, output_format, output_path)
                    self.report_generator.save_report(results, output_format)

            return results

        except Exception as e:
            logger.error(f"Error during EDA process: {e}")
            logger.error(traceback.format_exc())
            raise


    def print_basic_results(self, results:EDAResults, output_path:str):
        """Prints basic results to console"""

        print("="*50)
        print(f"\n A DATA ANALYSIS REPORT HAS BEEN CREATED AND SAVED TO {output_path}/eda_report.txt")
        # print(f"Shape: {results.shape}\n")
        # print(f"Columns: {results.columns}\n")
        # print(f"Columns with Nulls: {results.columns_with_null_values}\n")
        # print(f"High correlations: {len(results.store_corr_matrix)} pairs found:\n")

        # for x_col, y_col, corr in results.store_corr_matrix:
        #     print(f"{x_col}<->{y_col}: {corr:.2f}")

        # if results.datatype_issues:
        #     print(f"\n Potential type mismatches found in {len(results.datatype_issues)} columns:")
        #     for col, issues in results.datatype_issues.items():
        #         print(f"{col}: {','.join(issues)}")

# def main():
#     """Main function for CLI"""
    
#     parser = argparse.ArgumentParser(
#         description="Perform automatic EDA for your CSV files",
#         formatter_class=argparse.RawDescriptionHelpFormatter
#     )

#     parser.add_argument("--file_path", type=str, required=True, help="Path to the CSV file")
#     parser.add_argument("--file_type", type=str, required=True, help="File extension type(eg 'csv')")
#     parser.add_argument("--use_llm", action="store_true", help="Use local LM to provide EDA suggestions")
#     parser.add_argument("--visualize", action="store_true", help="Create visualizations for the data")
#     parser.add_argument("--save_report", action="store_true", help="Save the EDA output to file")
#     parser.add_argument("--output_format", choices=["txt","json"], default="txt", help="Output report format")
#     parser.add_argument("--model", type=str, default="llama3.2:latest", help="Ollama model to be used for analysis")
#     parser.add_argument("--output_path", type=str, default = "eda_report", help="path of directory where report is to be saved")
#     parser.add_argument("--plot_dir", type=str, default = "eda_plots", help="Directory to save visualisation plots")

#     args=parser.parse_args()

#     try:
#         if not args.file_path or not args.file_type:
#             raise ValueError("You either forgot to specify the file type or the file path.")

#         # INITIALISE AND RUN AUTO-EDA
#         auto_eda = AutoEDA(model_name = args.model, plot_output_dir = args.plot_dir)

#         results = auto_eda.run_eda(
#                     file_path = args.file_path,
#                     file_type = args.file_type,
#                     use_llm = args.use_llm,
#                     visualize = args.visualize,
#                     save_report = args.save_report,
#                     output_format = args.output_format,
#                     output_path = args.output_path,
#                     plot_output_dir = args.plot_dir
#                 )

#         auto_eda.print_basic_results(results)

#     except Exception as e:
#         logger.error("You did something wrong somewhere my bro:{e}")
#         logger.error(traceback.format_exc())

# if __name__ == "__main__":
#     main()
