from src.models import *
import pickle
import torch
from torch.utils.data import TensorDataset

import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')


CLASSIFICATION_MODEL_PATH = '/home/mikeyasnov/detecting_antibiotics/Pipeline/src/galacticum_cnn_milk_v1.pth'
REGRESSION_MODEL_PATH = '/home/mikeyasnov/detecting_antibiotics/Pipeline/src/cat_regression.pkl'
LABEL_ENCODER_PATH = '/home/mikeyasnov/detecting_antibiotics/Pipeline/src/le.sav'




class Pipeline():
    def __init__(self, data):
        self.class_model = ClfModelTabCNN(input_dim=15600,output_dim=3)
        self.class_model.load_state_dict(torch.load(CLASSIFICATION_MODEL_PATH, map_location='cpu'))
        self.class_model.eval()
        self.reg_model = pickle.load(open(REGRESSION_MODEL_PATH, 'rb'))
        self.label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))

        self.X = data

    def get_classification(self, ):
        X = torch.tensor(self.X, dtype=torch.float).reshape(1, 1040)
        logits = self.class_model(X)
        prediction = logits.argmax(1).numpy()

        decoded_prediction = self.label_encoder.inverse_transform(prediction)[0]
        return decoded_prediction

    def get_regression(self, ):
        return self.reg_model.predict(self.X)

    def get_predicted_data(self, ):
        data = {f'feature_{i}': x for (i, x) in enumerate(self.X)}
        data['substance'] = self.get_classification()
        data['target'] = self.get_regression()
        return data.values()


def main():
    df = pd.read_csv('/home/mikeyasnov/detecting_antibiotics/Pipeline/_current_voltage.csv').T.reset_index()
    data = np.array(df.iloc[1, :])
    data[0] = float(data[0])
    data = np.array(data, dtype=np.float)

    pipe = Pipeline(data=data)
    print(pipe.get_predicted_data())


if __name__ == '__main__':
    main()