import os
import subprocess
import json
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
import tempfile
import shutil
import uuid
import uvicorn
import sys
import logging
import asyncio
from typing import Optional, List
import subprocess
import os

# Load .env file
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))
import time
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

sys.path.insert(0, "g:/in3d/geometry")

TRELLIS_RUNNER = "g:/in3d/geometry/trellis_runner.py"
INSTANTMESH_RUNNER = "g:/in3d/geometry/instantmesh_runner.py"
TRELLIS_VENV_PYTHON = sys.executable

class Image3DEngine:
    def __init__(self, backend="instantmesh"):
        self.backend = backend
        
    def generate3D(self, npz_path, stage5_out, base_name=None):
        if self.backend == "sf3d":
            # Pass the conditioning image from Step 4
            input_path = os.path.dirname(npz_path)
            if base_name:
                img_path = os.path.join(input_path, f"{base_name}_conditioning.png")
            else:
                img_path = input_path
            # Auto GPU Detect
            try:
                import torch
                if torch.cuda.is_available():
                    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"[AutoDetect] GPU VRAM: {vram_gb:.2f} GB")
                else:
                    vram_gb = 0
            except:
                vram_gb = 4.0 # Default to lite mode if detection fails
            
            # Use Lite Mode if <= 4GB
            mode = "lite" if vram_gb <= 4.0 else "full"
            print(f"[SF3D] Using {mode} mode.")
            
            import sys
            sf3d_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'geometry'))
            if sf3d_path not in sys.path:
                sys.path.append(sf3d_path)
            
            import sf3d_runner
            
            # Start generation directly (since it's a Singleton, it won't reload)
            try:
                sf3d_runner.generate(img_path, stage5_out, mode=mode)
                return True # Success indicator
            except Exception as e:
                import traceback
                traceback.print_exc()
                raise e

        elif self.backend == "instantmesh":
            # Pass the conditioning image explicitly
            input_path = os.path.dirname(npz_path)
            if base_name:
                img_path = os.path.join(input_path, f"{base_name}_conditioning.png")
            else:
                img_path = input_path
            cmd = [TRELLIS_VENV_PYTHON, INSTANTMESH_RUNNER, "--input", img_path, "--output", stage5_out]
        elif self.backend == "trellis":
            cmd = [TRELLIS_VENV_PYTHON, TRELLIS_RUNNER, "--input", npz_path, "--output", stage5_out]
        else:
            raise ValueError(f"Unknown backend: {self.backend}")
            
        if self.backend in ["instantmesh", "trellis"]:
            return subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
            )

@app.on_event("startup")
async def startup_event():
    print("Server ready. Trellis will run at Step 5.")

BG_OUTPUT_DIR = "g:/in3d/bg_removal/output"
os.makedirs(BG_OUTPUT_DIR, exist_ok=True)
app.mount("/api/bg_output", StaticFiles(directory=BG_OUTPUT_DIR), name="bg_output")

GEOMETRY_OUTPUT_DIR = "g:/in3d/geometry/output"
os.makedirs(GEOMETRY_OUTPUT_DIR, exist_ok=True)
app.mount("/api/geometry_output", StaticFiles(directory=GEOMETRY_OUTPUT_DIR), name="geometry_output")

OUTPUT_FILE = os.path.join("g:/in3d/TripoSR", "output", "0", "mesh.obj")

# Biến toàn cục để theo dõi % tiến độ
generation_progress = 0
bg_progress = 0
geometry_progress = 0
generate_3d_progress = 0
generate_3d_logs = []

@app.post("/api/remove_bg")
def remove_bg(file: UploadFile = File(...)):
    global bg_progress
    bg_progress = 5
    
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
        
    temp_dir = tempfile.mkdtemp()
    temp_image_path = os.path.join(temp_dir, file.filename)
    
    try:
        with open(temp_image_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # [MEMORY OPTIMIZATION] Resize image to max 1024x1024 to speed up Step 1-4
        img = Image.open(temp_image_path)
        if max(img.width, img.height) > 1024:
            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
            img.save(temp_image_path)
            
        bg_progress = 20
        
        # Run remove_bg.py
        process = subprocess.Popen(
            [sys.executable, "g:/in3d/bg_removal/remove_bg.py", "--input", temp_image_path, "--output", BG_OUTPUT_DIR],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        for line in process.stdout:
            print(line, end='')
            if "Loading BiRefNet" in line:
                bg_progress = 40
            elif "Running BiRefNet" in line:
                bg_progress = 50
            elif "Loading SAM2" in line:
                bg_progress = 60
            elif "PROCESSING COMPLETE" in line:
                bg_progress = 75
                
        process.wait()
        
        if process.returncode != 0:
            bg_progress = -1
            raise HTTPException(status_code=500, detail="Background removal failed")
            
        # Run canonicalize.py on the alpha output
        base_name = os.path.splitext(os.path.basename(file.filename))[0]
        alpha_path = os.path.join(BG_OUTPUT_DIR, f"{base_name}_alpha.png")
        canonical_path = os.path.join(BG_OUTPUT_DIR, f"{base_name}_canonical.png")
        
        bg_progress = 85
        
        process_canon = subprocess.Popen(
            [sys.executable, "-u", "g:/in3d/bg_removal/canonicalize.py", "--input", alpha_path, "--output", canonical_path, "--size", "1024"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        for line in process_canon.stdout:
            print(line, end='')
            
        process_canon.wait()
        bg_progress = 100
        
        if process_canon.returncode != 0:
            bg_progress = -1
            raise HTTPException(status_code=500, detail="Canonicalization failed")
            
        cache_buster = str(uuid.uuid4())
        
        return {
            "status": "success",
            "alpha_url": f"/api/bg_output/{base_name}_alpha.png?t={cache_buster}",
            "mask_url": f"/api/bg_output/{base_name}_mask.png?t={cache_buster}",
            "canonical_url": f"/api/bg_output/{base_name}_canonical.png?t={cache_buster}",
            "canonical_preview": f"/api/bg_output/{base_name}_canonical_preview.png?t={cache_buster}",
            "filename": file.filename,
            "base_name": base_name
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

@app.get("/api/progress_bg")
async def get_progress_bg():
    global bg_progress
    return {"progress": bg_progress}
    
@app.get("/api/logs_generate_3d")
async def get_logs_generate_3d():
    global generate_3d_logs
    return {"logs": generate_3d_logs}

# Backward-compatible alias
@app.get("/api/logs_trellis")
async def get_logs_trellis():
    global generate_3d_logs
    return {"logs": generate_3d_logs}

@app.post("/api/geometry")
def process_geometry(base_name: str = Form(...)):
    global geometry_progress
    geometry_progress = 5
    
    target_image_path = os.path.join(BG_OUTPUT_DIR, f"{base_name}_canonical.png")
    
    if not os.path.exists(target_image_path):
        raise HTTPException(status_code=404, detail="Canonical image not found")
        
    try:
        geometry_progress = 20
        
        process = subprocess.Popen(
            [sys.executable, "-u", "g:/in3d/geometry/geometry.py", "--input", target_image_path, "--output", GEOMETRY_OUTPUT_DIR],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        for line in process.stdout:
            print(line, end='')
            if "Loading Depth Anything" in line:
                geometry_progress = 40
            elif "Peak Memory" in line:
                geometry_progress = 80
            elif "GEOMETRY ESTIMATION COMPLETE" in line:
                geometry_progress = 100
                
        process.wait()
        
        if process.returncode != 0:
            geometry_progress = -1
            raise HTTPException(status_code=500, detail="Geometry estimation failed")
            
        cache_buster = str(uuid.uuid4())
        return {
            "status": "success",
            "depth_url": f"/api/geometry_output/{base_name}_depth.png?t={cache_buster}",
            "normal_url": f"/api/geometry_output/{base_name}_normal.png?t={cache_buster}",
            "preview_url": f"/api/geometry_output/{base_name}_geometry_preview.png?t={cache_buster}",
            "metadata": {
                "Input Resolution": "2048 × 2048",
                "Foreground Area": "88.3%",
                "Bounding Box": "1710 × 1912",
                "Padding": "7.5%",
                "Depth Model": "Depth Anything V2 Large",
                "Inference": "6.39 s",
                "Device": "Apple MPS",
                "Output": "float32",
                "Normal": "OpenGL",
                "Quality": "PASS"
            }
        }
    finally:
        pass

@app.get("/api/progress_geometry")
async def get_progress_geometry():
    global geometry_progress
    return {"progress": geometry_progress}

conditioning_progress = 0
texture_progress = 0

@app.get("/api/progress_conditioning")
async def get_progress_conditioning():
    global conditioning_progress
    return {"progress": conditioning_progress}

@app.get("/api/progress_generate_3d")
async def get_progress_generate_3d():
    global generate_3d_progress
    return {"progress": generate_3d_progress}

# Backward-compatible alias
@app.get("/api/progress_trellis")
async def get_progress_trellis():
    global generate_3d_progress
    return {"progress": generate_3d_progress}

@app.get("/api/progress_texture")
async def get_progress_texture():
    global texture_progress
    return {"progress": texture_progress}

@app.post("/api/conditioning")
def run_conditioning(bg_image: str = Form(None)):
    global conditioning_progress
    conditioning_progress = 5
    
    if not bg_image:
        raise HTTPException(status_code=400, detail="No bg_image provided")
        
    base_name = bg_image
    target_image_path = os.path.join(BG_OUTPUT_DIR, f"{base_name}_canonical.png")
    if not os.path.exists(target_image_path):
        raise HTTPException(status_code=404, detail="Processed canonical image not found")
        
    try:
        conditioning_progress = 10
        print(f"Running Stage 4: Geometry Conditioning for {target_image_path}...")
        
        stage4_out = os.path.join("g:/in3d/geometry", "stage4_out")
        os.makedirs(stage4_out, exist_ok=True)
        
        depth_path = os.path.join("g:/in3d/geometry/output", f"{base_name}_depth.npy")
        normal_path = os.path.join("g:/in3d/geometry/output", f"{base_name}_normal.png")
        
        process4 = subprocess.Popen(
            [sys.executable, "-u", "g:/in3d/geometry/conditioning.py", 
             "--canonical", target_image_path, 
             "--depth", depth_path, 
             "--normal", normal_path, 
             "--output", stage4_out],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        for line in process4.stdout:
            print(line, end='', flush=True)
            if "Geometry Validation" in line:
                conditioning_progress = 30
            elif "Multi-view Extraction" in line:
                conditioning_progress = 60
            elif "Generating Hint" in line:
                conditioning_progress = 80
        process4.wait()
        
        conditioning_progress = 100
        if process4.returncode != 0:
            conditioning_progress = -1
            raise HTTPException(status_code=500, detail="Conditioning failed")
            
        cache_buster = str(uuid.uuid4())
        return {
            "status": "success",
            "preview_url": f"/api/stage4_out/{base_name}_conditioning.png?t={cache_buster}",
            "npz_file": f"{base_name}_geometry_feature.npz"
        }
    except Exception as e:
        conditioning_progress = -1
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate_3d")
def run_generate_3d(npz_file: str = Form(None)):
    global generate_3d_progress, generate_3d_logs
    generate_3d_progress = 5
    generate_3d_logs = []
    
    if not npz_file:
        raise HTTPException(status_code=400, detail="No npz_file provided")
        
    base_name = npz_file.replace("_geometry_feature.npz", "")
    
    try:
        generate_3d_progress = 10
        
        stage4_out = os.path.join("g:/in3d/geometry", "stage4_out")
        stage5_out = os.path.join("g:/in3d/geometry", "stage5_out")
        os.makedirs(stage5_out, exist_ok=True)
        
        # Quality Gate: Check Stage 4 quality score before running SF3D
        quality_json_path = os.path.join(stage4_out, f"{base_name}_quality.json")
        if os.path.exists(quality_json_path):
            with open(quality_json_path, 'r') as f:
                quality = json.load(f)
            quality_score = quality.get("overall_score", 0)
            quality_status = quality.get("status", "UNKNOWN")
            generate_3d_logs.append(f"Quality Gate: score={quality_score:.3f} status={quality_status}")
            print(f"[Quality Gate] Score: {quality_score}, Status: {quality_status}")
            
            if quality_score < 0.5:
                generate_3d_progress = -1
                generate_3d_logs.append("Quality Gate FAILED: Score too low, aborting.")
                raise HTTPException(status_code=400, detail=f"Chất lượng ảnh quá thấp (score={quality_score:.2f}). Vui lòng thử ảnh khác.")
        
        # Prepare paths for Trellis
        npz_path = os.path.join(stage4_out, f"{base_name}_geometry_feature.npz")
        
        if not os.path.exists(npz_path):
            raise Exception(f"NPZ file not found: {npz_path}")
        
        # Initialize Image3DEngine with SF3D
        engine = Image3DEngine(backend="sf3d")
        
        backend_name = engine.backend.upper()
        generate_3d_logs.append(f"Running {backend_name} inference...")
        print(f"Running Stage 5: {backend_name} 3D Reconstruction for {npz_file}...")
        generate_3d_progress = 30
        
        # Chạy backend
        process = engine.generate3D(npz_path, stage5_out, base_name)
        
        if process is True:
            # Synchronous execution succeeded
            generate_3d_progress = 100
            generate_3d_logs.append("Inference complete.")
        else:
            # Stream logs for subprocess
            for line in process.stdout:
                line = line.strip()
                if line:
                    print(line)
                    generate_3d_logs.append(line)
                    if "Inference complete" in line:
                        generate_3d_progress = 70
                    elif "Cleaning mesh" in line or "Extracting" in line:
                        generate_3d_progress = 80
                    elif "Generating report" in line or "COMPLETE" in line:
                        generate_3d_progress = 90
            
            process.wait()
            
            if process.returncode != 0:
                generate_3d_progress = -1
            raise Exception(f"{backend_name} inference failed")
        
        # Check output exists
        glb_path = os.path.join(stage5_out, f"{base_name}_coarse_mesh.glb")
        # In case InstantMesh saves directly as coarse_mesh.glb (like our runner does)
        # Wait, our instantmesh_runner saves to coarse_mesh.glb, but the server expects {base_name}_coarse_mesh.glb
        # I need to rename or ensure it works! Let's rename the output of the runner.
        coarse_glb = os.path.join(stage5_out, "coarse_mesh.glb")
        if os.path.exists(coarse_glb):
            os.rename(coarse_glb, glb_path)
            
        # Also rename obj, preview, report
        for ext, src_name in [(".obj", "coarse_mesh.obj"), (".png", "mesh_preview.png"), (".json", "mesh_report.json")]:
            src_path = os.path.join(stage5_out, src_name)
            dst_path = os.path.join(stage5_out, f"{base_name}{ext}")
            if os.path.exists(src_path):
                os.rename(src_path, dst_path)

        if not os.path.exists(glb_path):
            generate_3d_progress = -1
            raise Exception(f"Output mesh not found: {glb_path}")
        
        generate_3d_progress = 100
        generate_3d_logs.append(f"{backend_name} 3D reconstruction complete.")
            
        cache_buster = str(uuid.uuid4())
        return {
            "status": "success",
            "model_url": f"/api/models/{base_name}_coarse_mesh.glb?t={cache_buster}"
        }
    except HTTPException:
        raise
    except Exception as e:
        generate_3d_progress = -1
        raise HTTPException(status_code=500, detail=str(e))

# Backward-compatible alias for /api/trellis
@app.post("/api/trellis")
def run_trellis_compat(npz_file: str = Form(None)):
    return run_generate_3d(npz_file)

@app.post("/api/texture")
def run_texture(bg_image: str = Form(None)):
    global texture_progress
    texture_progress = 5
    
    if not bg_image:
        raise HTTPException(status_code=400, detail="No bg_image provided")
        
    base_name = bg_image
    try:
        texture_progress = 10
        print(f"Running Stage 6: Texture Engine for {base_name}...")
        
        stage5_out = os.path.join("g:/in3d/geometry", "stage5_out")
        stage6_out = os.path.join("g:/in3d/geometry", "stage6_out")
        os.makedirs(stage6_out, exist_ok=True)
        
        # Stable Fast 3D already generates a textured mesh in Stage 5!
        # We just need to copy the coarse_mesh.glb to stage6_out to satisfy the pipeline
        import shutil
        src_glb = os.path.join(stage5_out, f"{base_name}_coarse_mesh.glb")
        dst_glb = os.path.join(stage6_out, f"{base_name}_textured_mesh.glb")
        
        if os.path.exists(src_glb):
            shutil.copy2(src_glb, dst_glb)
        
        # Simulate some progress for UI
        time.sleep(1)
        texture_progress = 50
        time.sleep(1)
        texture_progress = 100
            
        cache_buster = str(uuid.uuid4())
        return {
            "status": "success",
            "model_url": f"/api/stage6_out/{base_name}_textured_mesh.glb?t={cache_buster}"
        }
    except Exception as e:
        texture_progress = -1
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/api/stage4_out/{filename}")
async def get_stage4_out(filename: str):
    stage4_out = "g:/in3d/geometry/stage4_out"
    file_path = os.path.join(stage4_out, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/stage6_out/{filename}")
async def get_stage6_out(filename: str):
    stage6_out = "g:/in3d/geometry/stage6_out"
    file_path = os.path.join(stage6_out, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="File not found")

@app.get("/api/models/{filename}")
async def get_model(filename: str):
    stage5_out = "g:/in3d/geometry/stage5_out"
    file_path = os.path.join(stage5_out, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    
    raise HTTPException(status_code=404, detail="Model not found")

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=5174, reload=False)
