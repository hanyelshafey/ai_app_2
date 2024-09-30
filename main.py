from typing import Union
import io
from fastapi import FastAPI,UploadFile
from PIL import Image
from model import vilt_pipline
app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def ask(text:str,image:UploadFile):
    content=image.file.read()
    image=Image.open(io.BytesIO(content))
    result=vilt_pipline(text,image)
    
    return {"answer": result}

