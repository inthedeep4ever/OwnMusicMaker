from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
import os, shutil, wave, struct

app = FastAPI()

UPLOAD_DIR = "uploads"
GENERATED_DIR = "generated"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

@app.post("/api/upload")
async def upload_wav(file: UploadFile):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status":"uploaded","file":file.filename}

@app.get("/api/generate")
def generate():
    out_path = os.path.join(GENERATED_DIR, "test_output.wav")

    # 1 seconde silent WAV
    framerate = 44100
    nframes = framerate
    with wave.open(out_path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(framerate)
        frames = b''.join(struct.pack('<h', 0) for _ in range(nframes))
        wf.writeframes(frames)

    return FileResponse(out_path, media_type="audio/wav")

@app.post("/api/upload")
async def upload_file(file: UploadFile):
    filepath = os.path.join(UPLOAD_DIR, file.filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return {"status": "uploaded", "file": file.filename}
