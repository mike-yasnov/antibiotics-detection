
from dependency_injector.wiring import Provide, inject
from fastapi import Depends

from src.containers.conainers import AppContainer
from src.routes.routers import router
from src.services.predictor import Predictor
from src.services.preprocess_utils import preprocess_data



@router.get('/predict')
@inject
def predict(
    path: str,
    service: Predictor = Depends(Provide[AppContainer.predictor]),
):
    data = preprocess_data(path=path)
    antibiotic, concentration = service.predict(data)

    return {'antibiotic': antibiotic, 'concentration': concentration} 
