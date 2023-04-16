import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/hello")
def root():
    """
    Простейшний ендпоинт, возращающий hello world
    :return:
    """
    return {"message": "Hello World"}

uvicorn.run(app, host='0.0.0.0')
