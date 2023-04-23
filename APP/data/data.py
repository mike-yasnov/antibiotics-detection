import pandas as pd
import numpy as np 


# class Ivium:
#     def __init__(self, fname):
#         self.fname = fname
#         self.data, self.voltage, self.current = self._read_data()
#         self.data_for_prediction = self._get_dataframe_for_model()

#     def _read_data(self):
#         data = pd.read_csv(self.fname, index_col=0)
#         voltage = data[data.columns[0]].values
#         current = data[data.columns[1]].values

#         if len(voltage) != 1040:
#             return None, None, None
#         else:
#             return data, voltage, current

#     def _get_dataframe_for_model(self):
#         model_dataframe = pd.DataFrame(
#             data=[self.current],
#             columns=self.voltage
#             )
#         return model_dataframe

class Ivium:
    def __init__(self, fname):
        self.fname = fname
        self.cycle_len = 1040
        self.current = None
        self.voltage = None
        self.file = self.read_file()
        self.current = self.get_current()
        self.voltage = self.get_voltage()
        self.data = pd.DataFrame(data={
            'column_0': self.voltage,
            'column_1': self.current
        }, dtype=np.float64)


    def read_file(self):
        data = []
        with open(self.fname, 'r') as f:
            for line in f.readlines():
                data.append(line.strip().split(' '))
            f.close()
        prepared_data = []
        for line in data:
            prepared_data.append([x for x in line if x])
        return prepared_data[1:]

    def get_current(self):
        current = []
        for value in self.file:
            current.append(value[2])
        return current

    def get_voltage(self):
        voltage = []
        for value in self.file:
            voltage.append(value[1])
        return voltage
    
    def get_data(self, ):
        return