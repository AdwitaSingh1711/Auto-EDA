import pandas as pd
import pathlib as Path
import random
import numpy as np

def create_test_data():
    Path("tests/data").mkdir(exists_ok=True)

    # data = {
    #     'age': [25, 42, 35, 60, 28],
    #     'bp': [120, 140, 130, 150, 110],
    #     'cholesterol': ['normal', 'high', 'normal', 'high', 'normal'],
    #     'outcome': [0, 1, 0, 1, 0]
    # }

    random.seed(42)
    np.random.seed(42)

    cholesterol_options = ['normal', 'high']
    outcome_options = [0, 1]
    n_rows = 100

    data = {
        'age': np.random.randint(18, 80, size=n_rows).tolist(),
        'bp': np.random.randint(90, 180, size=n_rows).tolist(),
        'cholesterol': np.random.choice(cholesterol_options, size=n_rows).tolist(),
        'outcome': np.random.choice(outcome_options, size=n_rows).tolist()
    }

    for i in range(6):
        col_name = f'feature_{i+1}'
        data[col_name] = np.random.uniform(0, 1, size=n_rows).round(3).tolist()

    df = pd.DataFrame(data)
    df.to_csv("tests/data/test_Data.csv", index = False)

if __name__ = "__main__":
    create_test_data()