import numpy as np
import pandas as pd


def preprocess_data(path: str) -> np.array:
    return np.array(pd.read_csv(path, index_col=0).T.iloc[1])
