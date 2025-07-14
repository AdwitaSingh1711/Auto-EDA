# DivePy 

**DivePy** is a command-line Python tool for developers and data analysts. It performs automatic Exploratory Data Analysis (EDA) on CSV files. It summarizes the dataset, identifies missing values, creates elementary visualisation plots, highlights correlations, and can even suggest next steps using an LLM (via [Ollama](https://ollama.com)).

---
## Demo

https://github.com/user-attachments/assets/34df7469-198a-4b38-ac6a-5b53bf4cbad1

---

## Installation

### 1. Install Ollama from [here](https://ollama.com/)

Make sure to run the ollama server in a separate terminal window if you're setting --use__llm = True for generating recommendations using:

```bash
ollama serve
```

### 2. Usage
For checking the list of valid arguments, use:
```bash
divepy --help
```

For general usage without Ollama, use:
```bash
divepy --file_path path/to/data.csv --file_type csv --visualise True --visualize --save_report
```

---

## Features

- Automatic EDA for `.csv` files
- Correlation detection and plotting
- Optional LLM-powered insights (via Ollama)
- Outputs key dataset information:
  - Dataset shape
  - Column names
  - Null value counts
  - Data types and structure
  - Highly correlated columns
- Built-in plotting with Plotly (interactable HTML dashboard for visualisation)

---

## Contributing 

Still a work in Progress (pull requests are welcome though). If you have feature ideas or spot a bug, open an issue or fork and submit a PR [here](https://github.com/AdwitaSingh1711/Auto-EDA).
