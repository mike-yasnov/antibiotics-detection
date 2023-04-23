
from dependency_injector.wiring import Provide, inject
from fastapi import Depends, File
from fastapi.responses import HTMLResponse

from src.containers.conainers import AppContainer
from src.routes.routers import router
from src.services.predictor import Predictor
from src.services.preprocess_utils import preprocess_file



@router.post('/predict')
@inject
def predict(
    file: bytes = File(),
    service: Predictor = Depends(Provide[AppContainer.predictor]),
):
    data = preprocess_file(file=file)
    antibiotic, concentration = service.predict(data)

    return {'antibiotic': antibiotic, 'concentration': concentration} 

