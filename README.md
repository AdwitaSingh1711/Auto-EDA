# AutoEDA 

**AutoEDA** is a command-line Python tool for developers and data analysts that performs automatic Exploratory Data Analysis (EDA) on CSV files. It summarizes the dataset, identifies missing values, highlights correlations, and can even suggest next steps using an LLM (via [Ollama](https://ollama.com)).

---

## Features

- Automatic EDA for `.csv` files
- Correlation detection and plotting (scatter + pie charts)
- Optional LLM-powered insights (via Ollama)
- Outputs key dataset information:
  - Dataset shape
  - Column names
  - Null value counts
  - Data types and structure
  - Highly correlated columns
- Built-in plotting with Plotly

---

## Installation

### 1. Install Ollama from [here](https://ollama.com/)

### 2. Clone the repo

```bash
git clone https://github.com/yourusername/autoeda.git
cd autoeda

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Usage
```
python autoeda.py --file_path path/to/data.csv --file_type csv --visualise True --use_llm True
```

## Contributing 

Pull requests are welcome! If you have feature ideas or spot a bug, open an issue or fork and submit a PR.