from typing import List, Dict

import numpy as np
from src.services.classifier import DataClassfifier
from src.services.regressor import DataRegressor        


class Predictor:
    def __init__(self, regressor: DataRegressor, classifier: DataClassfifier):
        self._regressor = regressor
        self._classifier = classifier

    def predict(self, data: np.ndarray) -> List:
        antibiotic = self._classifier.predict(data=data)[0]
        if antibiotic != 'milk':
            quantity = self._regressor.predict(data=data)
        else:
            quantity = 0.0
        return [antibiotic, quantity]
