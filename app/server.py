from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
import os, shutil, uuid, json
from dataset import SamplePackCatalog, SR
import torchaudio
import asyncio
from pathlib import Path

from generate import generate_mix

# Define paths
UPLOAD_DIR = "uploads"
PACKS_DIR = "packs"
GENERATED_DIR = "generated"

# Create directories
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(PACKS_DIR, exist_ok=True)
os.makedirs(GENERATED_DIR, exist_ok=True)

app = FastAPI()
catalog = SamplePackCatalog()

# Track generation status
_generation_status = {}

@app.get("/")
async def root():
    return {"message": "OwnMusicMaker API", "status": "running"}

@app.post("/api/upload")
async def upload_file(file: UploadFile = File(...), pack: str = Form(None), category: str = Form(None)):
    try:
        filename = file.filename
        save_dir = UPLOAD_DIR
        
        if pack and category:
            save_dir = os.path.join(PACKS_DIR, pack, category)
        
        os.makedirs(save_dir, exist_ok=True)
        filepath = os.path.join(save_dir, filename)
        
        # Save file
        contents = await file.read()
        with open(filepath, "wb") as f:
            f.write(contents)
        
        # Preprocess: resample to project SR
        try:
            wav, orig_sr = torchaudio.load(filepath)
            if orig_sr != SR:
                resampler = torchaudio.transforms.Resample(orig_sr, SR)
                wav = resampler(wav)
                torchaudio.save(filepath, wav, SR)
                print(f"Resampled {filename} from {orig_sr} to {SR}")
        except Exception as e:
            print(f"Resample warning for {filename}: {e}")
            # Continue anyway - file still usable
        
        catalog._build()
        return {"status": "ok", "file": filename, "path": filepath}
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

def _do_generate_async(job_id, sample_list, out_path, duration_sec):
    """Background task to generate mix"""
    try:
        _generation_status[job_id] = {"status": "generating", "progress": 0}
        print(f"[{job_id}] Starting generation with {len(sample_list)} samples, {duration_sec}s duration")
        
        generate_mix(sample_list, out_path=out_path, duration_sec=duration_sec)
        
        _generation_status[job_id] = {"status": "done", "file": os.path.basename(out_path)}
        print(f"[{job_id}] Generation complete: {out_path}")
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        _generation_status[job_id] = {"status": "error", "error": str(e)}
        print(f"[{job_id}] Generation failed: {e}")

@app.post("/api/mix")
async def api_mix(selection: str = Form(...), duration_sec: int = Form(60), out_name: str = Form(None), background_tasks: BackgroundTasks = None):
    try:
        sample_list = json.loads(selection)
        
        # Validate inputs
        if not isinstance(sample_list, list) or len(sample_list) == 0:
            return JSONResponse({"error": "selection must be non-empty list"}, status_code=400)
        
        if duration_sec < 1 or duration_sec > 600:
            return JSONResponse({"error": "duration_sec must be 1-600"}, status_code=400)
        
        # Validate all paths exist
        real_paths = []
        for p in sample_list:
            real_path = p if os.path.isabs(p) else os.path.join(PACKS_DIR, p)
            if not os.path.exists(real_path):
                return JSONResponse({"error": f"File not found: {p}"}, status_code=404)
            real_paths.append(real_path)
        
        # Generate output filename
        out_name = out_name or f"mix_{uuid.uuid4().hex[:8]}.wav"
        out_path = os.path.join(GENERATED_DIR, out_name)
        job_id = uuid.uuid4().hex[:8]
        
        # Schedule background task
        background_tasks.add_task(_do_generate_async, job_id, real_paths, out_path, duration_sec)
        
        return {
            "status": "scheduled",
            "job_id": job_id,
            "file": os.path.basename(out_path),
            "url": f"/generated/{os.path.basename(out_path)}"
        }
    
    except json.JSONDecodeError:
        return JSONResponse({"error": "invalid JSON in selection"}, status_code=400)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Check generation status"""
    status = _generation_status.get(job_id)
    if status is None:
        return JSONResponse({"error": "job not found"}, status_code=404)
    return status

@app.get("/generated/{filename}")
async def get_generated(filename: str):
    """Download generated WAV"""
    filepath = os.path.join(GENERATED_DIR, filename)
    if not os.path.exists(filepath):
        return JSONResponse({"error": "file not found"}, status_code=404)
    return FileResponse(filepath, media_type="audio/wav")

@app.get("/api/catalog")
async def get_catalog():
    """List all available samples"""
    return catalog.to_dict()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)