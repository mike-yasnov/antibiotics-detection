import pickle
import torch
import warnings
from app.pipeline.src.models import ClfModelTabCNN

warnings.filterwarnings('ignore')

CLASSIFICATION_MODELS_DICT = dict(
    CNN=ClfModelTabCNN.load_from_checkpoint(
        'app/pipeline/src/cnn_ivium.ckpt',
        input_dim=1040,
        output_dim=6
        ),
    Cat=pickle.load(open('app/pipeline/src/cat_ivium.pkl', 'rb')),
    Lama=None,
    Fedot=None
)

REGRESSION_MODELS_DICT = dict(
    Cat=pickle.load(open('app/pipeline/src/cat_regression.pkl', 'rb')),
    Lama=None,
    Fedot=pickle.load(open('app/pipeline/src/linreg.pkl', 'rb'))
)

LABEL_ENCODER_PATH = 'app/pipeline/src/le.sav'


class Pipeline:
    def __init__(self, data, claffifier_type='CNN', regressor_type='Cat'):
        """
        Args:
        data: data for prediction
        classifier_type: Classification model name (CNN, Cat (Catboost),
        Lama (LightAutoMl), Fedot (AutoMl))
        regressor_type: Regression mode name (Cat (Catboost),
        Lama (LightAutoMl), Fedot (AutoMl))
        """

        self.class_model_name = claffifier_type
        self.reg_model_name = regressor_type

        self.class_model = CLASSIFICATION_MODELS_DICT[claffifier_type]

        self.reg_model = REGRESSION_MODELS_DICT[regressor_type]
        self.label_encoder = pickle.load(open(LABEL_ENCODER_PATH, 'rb'))

        self.X = data

    def get_classification(self, ):
        if self.class_model_name == 'CNN':
            self.class_model.eval()
            x = torch.tensor(self.X, dtype=torch.float).reshape(1, -1)
            logits = self.class_model(x)
            prediction = logits.argmax(1).numpy()

        elif self.class_model_name in ['Cat', 'Lama', 'Fedot']:
            prediction = self.class_model.predict(self.X.reshape(1, -1))
        else:
            raise Exception(
                'No pretrained model for this type. Please check the name'
                )
        decoded_prediction = self.label_encoder.inverse_transform(
            prediction
            )[0]
        return decoded_prediction

    def get_regression(self, ):
        x = self.X.reshape(1, -1)
        return self.reg_model.predict(x)

    def get_predicted_data(self, ):
        data = {f'feature_{i}': x for (i, x) in enumerate(self.X)}
        data['substance'] = self.get_classification()
        data['target'] = self.get_regression()
        return data.values()
