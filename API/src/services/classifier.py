import typing as tp

import numpy as np
import pickle 
import torch
from src.services.model import ClfModelTabCNN   

class DataClassfifier:
    def __init__(self, config: tp.Dict):
        self._label_encoder = pickle.load(
            open(config['label_encoder'], 'rb')
            )

        self._model = ClfModelTabCNN.load_from_checkpoint(
            config['model_path'],
            input_dim=1040,
            output_dim=6
            )
        self._classes= self._label_encoder.classes_

    @property
    def classes(self) -> tp.List:
        return list(self._classes)

    def predict(self, data: np.ndarray) -> float:
        self._model.eval()

        x = torch.tensor(data, dtype=torch.float).reshape(1, -1)
        logits = self._model(x)
        predictions = logits.argmax(1).numpy()
        return self._label_encoder.inverse_transform(predictions).tolist()
