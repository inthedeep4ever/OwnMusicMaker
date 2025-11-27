from fastapi import FastAPI, UploadFile
import shutil
import os

app = FastAPI()

UPLOAD_DIR = "uploads"
GENERATED_DIR = "generated"

@app.post("/api/upload")
def upload_wav(file: UploadFile):
    # Let op 4 spaties inspringing
    filepath = f"{UPLOAD_DIR}/{file.filename}"
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"status": "uploaded", "file": file.filename}


@app.get("/api/generate")
def generate():
    # Alles ingesprongen binnen de functie
    os.system("python generate.py")
    return {"status": "done", "file": f"{GENERATED_DIR}/output.wav"}