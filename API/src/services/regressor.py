from typing import Dict

import numpy as np
import pickle



class DataRegressor:

    def __init__(self, config: Dict):
        self._model = pickle.load(open(config['model_path'], 'rb'))


    def predict(self, data: np.ndarray) -> float:
        x = data.reshape(1, -1)
        print(x.shape)
        return self._model.predict(x)[0]

