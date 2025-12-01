from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import os, shutil, uuid
from dataset import SamplePackCatalog, load_mono, SR
from generate import generate_mix
import json

app = FastAPI()
UPLOAD_DIR = "uploads"
GENERATED_DIR = "generated"
PACKS_DIR = "packs"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)
os.makedirs(PACKS_DIR, exist_ok=True)

catalog = SamplePackCatalog(PACKS_DIR)

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), pack: str = Form(None), category: str = Form(None)):
    filename = file.filename
    save_dir = UPLOAD_DIR
    if pack and category:
        # save directly into packs structure if pack/category provided
        save_dir = os.path.join(PACKS_DIR, pack, category)
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    # update catalog
    catalog._build()
    return {"status": "ok", "file": filename, "path": filepath}

@app.get("/api/catalog")
def api_catalog():
    return catalog.catalog

@app.post("/api/mix")
async def api_mix(selection: str = Form(...), duration_sec: int = Form(60), out_name: str = Form(None)):
    """
    selection: JSON string list of sample file paths relative to packs/, or full paths
    Example: '["packs/pack1/loops/loop1.wav","packs/pack2/claps/clap3.wav"]'
    """
    try:
        sample_list = json.loads(selection)
    except Exception as e:
        return JSONResponse({"error": "invalid selection json", "detail": str(e)}, status_code=400)

    # sanitize paths if relative
    real_paths = []
    for p in sample_list:
        if p.startswith("packs/") or os.path.isabs(p):
            real_paths.append(p)
        else:
            real_paths.append(os.path.join(PACKS_DIR, p))

    out_name = out_name or f"mix_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(GENERATED_DIR, out_name)
    # generate synchronously (can be slow for long durations)
    generated = generate_mix(real_paths, out_path=out_path, duration_sec=duration_sec)
    return {"status": "ok", "file": os.path.basename(generated), "path": generated, "url": f"/generated/{os.path.basename(generated)}"}

@app.get("/generated/{filename}")
def serve_generated(filename: str):
    path = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")