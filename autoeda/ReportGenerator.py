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
from .EDAResults import EDAResults

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