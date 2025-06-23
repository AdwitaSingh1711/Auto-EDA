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
                # print(f"Test successful! Response: {response['message']['content']}")
                print(f"Test successful!")
                return True
            else:
                print("Empty test response")
                return False
        except Exception as e:
            print(f"Test failed: {e}")
            return False