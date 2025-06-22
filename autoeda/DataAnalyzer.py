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


class DataAnalyzer:
    """Data cleaning and preprocessing"""
    def __init__(self, max_unique_threshold: int = 10):
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