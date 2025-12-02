from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import os, shutil, uuid, json
from dataset import SamplePackCatalog

from generate import generate_mix

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
        save_dir = os.path.join(PACKS_DIR, pack, category)
        os.makedirs(save_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    with open(filepath, "wb") as f:
        shutil.copyfileobj(file.file, f)
    catalog._build()
    return {"status": "ok", "file": filename, "path": filepath}

@app.get("/api/catalog")
def api_catalog():
    return catalog.catalog

@app.post("/api/mix")
async def api_mix(selection: str = Form(...), duration_sec: int = Form(60), out_name: str = Form(None)):
    try:
        sample_list = json.loads(selection)
    except:
        return JSONResponse({"error":"invalid json"}, status_code=400)
    real_paths = [p if os.path.isabs(p) else os.path.join(PACKS_DIR,p) for p in sample_list]
    out_name = out_name or f"mix_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(GENERATED_DIR, out_name)
    generate_mix(real_paths, out_path=out_path, duration_sec=duration_sec)
    return {"status": "ok", "file": os.path.basename(out_path), "url": f"/generated/{os.path.basename(out_path)}"}

@app.get("/generated/{filename}")
def serve_generated(filename: str):
    path = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse({"error":"not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


#___________________________________________
# filepath: [server.py](http://_vscodecontentref_/10)
# ...existing code...
from fastapi import BackgroundTasks
# ...existing code...

def _do_generate_async(sample_list, out_path, duration_sec):
    try:
        generate_mix(sample_list, out_path=out_path, duration_sec=duration_sec)
    except Exception:
        # log error; don't crash server
        import traceback; traceback.print_exc()

@app.post("/api/mix")
async def api_mix(selection: str = Form(...), duration_sec: int = Form(60), out_name: str = Form(None), background_tasks: BackgroundTasks = None):
    try:
        sample_list = json.loads(selection)
    except:
        return JSONResponse({"error":"invalid json"}, status_code=400)
    real_paths = [p if os.path.isabs(p) else os.path.join(PACKS_DIR,p) for p in sample_list]
    out_name = out_name or f"mix_{uuid.uuid4().hex[:8]}.wav"
    out_path = os.path.join(GENERATED_DIR, out_name)
    # schedule background job
    background_tasks.add_task(_do_generate_async, real_paths, out_path, duration_sec)
    return {"status": "scheduled", "file": os.path.basename(out_path), "url": f"/generated/{os.path.basename(out_path)}"}
# ...existing code...