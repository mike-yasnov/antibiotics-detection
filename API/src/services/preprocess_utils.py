import numpy as np
import pandas as pd
from io import BytesIO


def preprocess_data(path: str) -> np.array:
    return np.array(pd.read_csv(path, index_col=0).T.iloc[1])

def preprocess_file(file: bytes) -> np.array:
    return np.array(pd.read_csv(BytesIO(file), index_col=0).T.iloc[1])