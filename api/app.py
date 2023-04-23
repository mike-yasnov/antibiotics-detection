import uvicorn
from fastapi import FastAPI
from omegaconf import OmegaConf

from src.containers.conainers import AppContainer
from src.routes.routers import router as app_router
from src.routes import antibiotics


def create_app() -> FastAPI:
    container = AppContainer()
    cfg = OmegaConf.load('config/config.yml')
    container.config.from_dict(cfg)
    container.wire([antibiotics])

    app = FastAPI()
    app.include_router(app_router, prefix='/antibiotics', tags=['antibiotics'])
    return app


app = create_app()

if __name__ == '__main__':
    uvicorn.run(app, port=5000, host='0.0.0.0')