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
