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

    
class DataVisualizer:
    """Performs elementary data visualisation"""

    def __init__(self, output_dir: str="eda_plots"):
        self.output_dir = Path(output_dir)
        # self.output_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.created_files = []

    @staticmethod
    def safe_filename(text: str)->str:
        """Convert text to safe filename"""
        return "".join(c for c in text if c.isalnum() or c in (' ', '_','-')).rstrip()
    
    def open_report_in_browser(self, index_path: Path):
        """Opens HTML dashboard in default browser"""

        if not index_path and not index_path.exists():
            logger.warning("No index file found to open")
            return

        try:
            # Handle WSL
            if 'microsoft' in platform.uname().release.lower():
                wsl_path = subprocess.check_output(
                    ['wslpath','w',str(index_path.resolve())],
                    text=True
                ).strip()

                webbrowser.open(f"file://{wsl_path}")
            else:
                webbrowser.open(f"file://{index_path.resolve()}")
                
            logger.info(f"Opened report in browser: {index_path}")
            
        except Exception as e:
            logger.error(f"Could not open report: {e}")
        
        # else:
        #     logger.warning("No index file found to open")


    def create_scatter_plot(self, correlations: List[Tuple[str,str,float]], df:pd.DataFrame):
        """creates scatter plots for correlated variables"""
        if len(correlations)>=1:
            for i, (x_col, y_col, corr_eval) in enumerate(correlations):
                try:
                    fig= px.scatter(
                        df, x_col, y_col,
                        title=f"Scatter Plot: {x_col} vs {y_col} (r={corr_eval:.2f})"
                    )
                    # fig.show()

                    # SAVE FOR HTML
                    filename = f"Scatter_{DataVisualizer.safe_filename(x_col)}_vs_{DataVisualizer.safe_filename(y_col)}.html"
                    filepath =  self.output_dir/ filename
                    fig.write_html(str(filepath)
                    ,include_plotlyjs=True,
                    full_html=True )
                    self.created_files.append(filepath)

                    # # SAVE AS PNG
                    # png_filename = f"Scatter_{DataVisualizer.safe_filename(x_col)}_vs_{DataVisualizer.safe_filename(y_col)}.png"
                    # png_filepath = self.output_dir/png_filename
                    # fig.write_image(str(png_filepath), width = 800, height = 600)
                    # self.created_files.append(png_filepath)
                
                except Exception as e:
                    logger.warning(f"Could not create scatter plot for {x_col} vs {y_col}: {e}")


    def create_distribution_plots(self, df: pd.DataFrame):
        """Create distribution plots for numeric columns"""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            try:
                # Histogram
                fig_hist = px.histogram(df, x=col, title=f'Distribution of {col}')
                # fig_hist.show()

                hist_filename = f"histogram_{DataVisualizer.safe_filename(col)}.html"
                hist_filepath = self.output_dir/hist_filename
                fig_hist.write_html(str(hist_filepath),
                                    include_plotlyjs=True,
                                    full_html=True )
                self.created_files.append(hist_filepath)

                # png_hist_filename = f"Scatter_{DataVisualizer.safe_filename(col)}.png"
                # png_filepath = self.output_dir/png_hist_filename
                # fig_hist.write_image(str(png_hist_filepath), width = 800, height = 600)
                # self.created_files.append(png_filepath)
                
                # Box plot
                fig_box = px.box(df, y=col, title=f'Box Plot of {col}')
                box_filename = f"boxplot_{DataVisualizer.safe_filename(col)}.html"
                box_filepath = self.output_dir/box_filename
                fig_box.write_html(str(box_filepath),
                                    include_plotlyjs=True,
                                    full_html=True)
                self.created_files.append(box_filepath)

                # png_box_filename = f"Scatter_{DataVisualizer.safe_filename(x_col)}.png"
                # png_box_filepath = self.output_dir/png_box_filename
                # fig_box.write_image(str(png_box_filepath), width = 800, height = 600)
                # self.created_files.append(png_box_filepath)

                logger.info(f"Distribution plots saved for {col}")

                # fig_box.show()
            except Exception as e:
                logger.warning(f"Could not create distribution plots for {col}: {e}")


    def create_correlation_heatmap(self, df:pd.DataFrame):
        """Create an interactive correlation heatmap"""
        numeric_df = df.select_dtypes(include=[np.number])

        if not numeric_df.empty:
            try:
                corr_matrix = numeric_df.corr()
                fig = px.imshow(corr_matrix,
                            title="Correlation Heatmap",
                            color_continuous_scale='RdBu_r',
                            aspect="auto")

                # ADD TEXT CONNOCATIONS
                fig.update_traces(texttemplate="%{z:.2f}", textfont_size=10)

                corrmap_filename = "Correlation_heatmap.html"
                corrmap_filepath = self.output_dir/corrmap_filename
                fig.write_html(str(corrmap_filepath),
                                    include_plotlyjs=True,
                                    full_html=True)
                self.created_files.append(corrmap_filepath)

                # png_corrmap_filename = f"Correlation_heatmap.png"
                # png_corrmap_filepath = self.output_dir/png_corrmap_filename
                # fig.write_image(str(png_corrmap_filepath), width = 800, height = 600)
                # self.created_files.append(png_corrmap_filepath)

                logger.info("Correlation heatmap saved at {corrmap_filepath}")
                # fig.show()
            except Exception as e:
                logger.warning(f"Could not create correlation heatmap: {e}")


    def plot_categorical_distributions(self, df: pd.DataFrame, categorical_analysist:dict):
        """Plot distributions for categorical columns"""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns

        try:
            if categorical_cols:
                for col in categorical_cols:
                    if df[col].nunique() <= 20:  # Only plot if not too many categories
                        value_counts = df[col].value_counts()
                        fig = px.bar(x=value_counts.index, y=value_counts.values,
                                    title=f'Distribution of {col}')
                            # fig.show()

                        cat_filename = f"Categorical_plots{DataVisualizer.safe_filename(col)}.html"
                        cat_filepath = self.output_dir/cat_filename
                        fig.write_html(str(cat_filepath), include_plotlyjs=True, full_html=True)
                        self.created_files.append(cat_filepath)

                        logger.info("Categorical plot saved to: {cat_filepath}")
            
            else:
                for col_name, col_data in categorical_analysist.items():
                    top_values = col_data['top_values']
                    categories = list(top_values.keys())
                    counts = list(top_values.values())

                    str_categories = [str(c) for c in categories]
                    fig = px.bar(
                        x=str_categories,
                        y=counts,
                        title = f"Disctribution of {col_name}",
                        labels = {'x': col_name, 'y':'Count'}
                    )

                    fig.update_layout(
                        xaxis_tickagnle =-45, hovermode = 'x'
                    )

                    cat_filename = f"Categorical_plots{DataVisualizer.safe_filename(col_name)}.html"
                    cat_filepath = self.output_dir/cat_filename
                    fig.write_html(str(cat_filepath),include_plotlyjs=True,full_html=True)
                    self.created_files.append(cat_filepath)

                    # png_cat_filename = f"Categorical_plots_{DataVisualizer.safe_filename(col)}.png"
                    # png_cat_filepath = self.output_dir/png_cat_filename
                    # fig.write_image(str(png_cat_filepath), width = 800, height = 600)
                    # self.created_files.append(png_cat_filepath)

                    logger.info(f"Categorical plot saved to: {cat_filepath}")
                        
        except Exception as e:
            logger.warning(f"Could not create categorical plot: {e}")
    
    def create_summary(self, df: pd.DataFrame, results):
        """HTML for capturing all visualisations"""
    

    def generate_plot_index(self):
        """HTML index for listing all visualisations"""

        try:
            html_content = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>EDA Visualization Report</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 40px; }
                    h1 { color: #333; }
                    .plot-link { 
                        display: block; 
                        margin: 10px 0; 
                        padding: 10px; 
                        background: #f5f5f5; 
                        text-decoration: none; 
                        border-radius: 5px;
                        color: #333;
                    }
                    .plot-link:hover { background: #e0e0e0; }
                </style>
            </head>
            <body>
                <h1>EDA Visualization Report</h1>
                <p>Click on the links below to view individual visualizations:</p>
            """

            for file_path in self.created_files:
                if file_path.suffix == '.html':
                    html_content += f'<a href="{file_path.name}" class="plot-link">{file_path.stem.replace("_", " ").title()}</a>\n'
            
            html_content += """\n</body></html>"""

            index_path = self.output_dir/"index.html"
            with open(index_path, 'w') as f:
                f.write(html_content)

            logger.success(f"Plot index created: {index_path}")

            return index_path
        
        except Exception as e:
            logger.error(f"Could not create plot index: {e}")
            return None

    
    def print_visualization_summary(self):
        """view summary of visualisations"""
        if self.created_files:
            print(f"\n Visualisations Created({len(self.created_files)} files)\n")
            print(f"Output directory: {self.output_dir.absolute()}\n")

            print("="*100)
            html_files = [f for f in self.created_files if f.suffix == '.html']
            png_files = [f for f in self.created_files if f.suffix == '.png']

            if html_files:
                print("\nInteractive file for visualisations\n")

                for file_path in html_files:
                    print(f"{file_path.name}")

            if png_files:
                print("\nStatic PNGs\n")

                for file_path in png_files:
                    print(f"{file_path.name}")

            print(f"\n To view the plots: open {self.output_dir}/index.html in your browser\n")

            print("="*100)

        else: 
            print("\nNo visualisations were created")