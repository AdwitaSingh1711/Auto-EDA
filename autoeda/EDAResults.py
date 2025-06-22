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