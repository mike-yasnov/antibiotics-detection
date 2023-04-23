import numpy as np
import pandas as pd
from typing import List
from io import BytesIO


class Ivium:
    def __init__(self, file: bytes) -> None:
        self.bytes_file = BytesIO(file)
        self.cycle_len = 1040
        self.current = None
        self.voltage = None
        self.file = self.read_file()
        self.current = self.get_current()
        self.voltage = self.get_voltage()
        self.data = pd.DataFrame(
            data={
            'column_0': self.voltage,
            'column_1': self.current
            }, 
            dtype=np.float64
            )


    def read_file(self) -> List:
        data = []
        for line in self.bytes_file:
            data.append(str(line).strip().split(' '))
        prepared_data = []
        for line in data:
            prepared_data.append([x for x in line if x])
        return prepared_data[1:]

    def get_current(self) -> List:
        current = []
        for value in self.file:
            current.append(value[2])
        return current

    def get_voltage(self) -> List:
        voltage = []
        for value in self.file:
            voltage.append(value[1])
        return voltage
     
    

def preprocess_data(path: str) -> np.array:
    return np.array(pd.read_csv(path, index_col=0).T.iloc[1])

def preprocess_file(file: bytes) -> np.array:
    return np.array(Ivium(file).data.T.iloc[1])