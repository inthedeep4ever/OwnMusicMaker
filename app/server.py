# ...existing code...
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
import os, shutil, uuid, json
from dataset import SamplePackCatalog, SR
import torchaudio

from generate import generate_mix

app = FastAPI()
# ...existing code...

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

    # preprocess: resample to project SR to avoid re-resampling later
    try:
        wav, orig_sr = torchaudio.load(filepath)
        if orig_sr != SR:
            wav = torchaudio.functional.resample(wav, orig_sr, SR)
            torchaudio.save(filepath, wav, SR)
    except Exception:
        # if resample fails, continue but log (don't crash upload)
        import traceback; traceback.print_exc()

    catalog._build()
    return {"status": "ok", "file": filename, "path": filepath}

def _do_generate_async(sample_list, out_path, duration_sec):
    try:
        generate_mix(sample_list, out_path=out_path, duration_sec=duration_sec)
    except Exception:
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
    background_tasks.add_task(_do_generate_async, real_paths, out_path, duration_sec)
    return {"status": "scheduled", "file": os.path.basename(out_path), "url": f"/generated/{os.path.basename(out_path)}"}
# ...existing code...