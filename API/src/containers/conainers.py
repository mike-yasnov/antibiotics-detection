from dependency_injector import containers, providers

from src.services.predictor import Predictor
from src.services.classifier import DataClassfifier
from src.services.regressor import DataRegressor


class AppContainer(containers.DeclarativeContainer):
    config = providers.Configuration()

    classifier = providers.Singleton(
        DataClassfifier,
        config=config.services.data_classifier,
    )

    regressor = providers.Singleton(
        DataRegressor,
        config=config.services.data_regressor,
    )

    predictor = providers.Singleton(
        Predictor,
        classifier=classifier,
        regressor=regressor,
    )
