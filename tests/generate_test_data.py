import pandas as pd
import pathlib as Path

def create_test_data():
    Path("tests/data").mkdir(exists_ok=True)

    data = {
        'age': [25, 42, 35, 60, 28],
        'bp': [120, 140, 130, 150, 110],
        'cholesterol': ['normal', 'high', 'normal', 'high', 'normal'],
        'outcome': [0, 1, 0, 1, 0]
    }

    df = pd.DataFrame(data)
    df.to_csv("tests/data/test_Data.csv", index = False)

if __name__ = "__main__":
    create_test_data()