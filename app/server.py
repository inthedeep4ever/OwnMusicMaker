from fastapi import FastAPI, UploadFile
import shutil
import os
from generate import *


app = FastAPI()


UPLOAD_DIR = "uploads"
GENERATED_DIR = "generated"


@app.post("/api/upload")
def upload_wav(file: UploadFile):
filepath = f"{UPLOAD_DIR}/{file.filename}"
with open(filepath, "wb") as buffer:
shutil.copyfileobj(file.file, buffer)
return {"status": "uploaded", "file": file.filename}


@app.get("/api/generate")
def generate():
os.system("python generate.py")
return {"status": "done", "file": "generated/output.wav"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000)
