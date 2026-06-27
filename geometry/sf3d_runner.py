import os
import sys
import torch
from contextlib import nullcontext
from PIL import Image
import trimesh
import numpy as np

# Ensure stable-fast-3d is in path
sf3d_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'stable-fast-3d'))
if sf3d_repo not in sys.path:
    sys.path.append(sf3d_repo)

import rembg
from sf3d.system import SF3D
from sf3d.utils import get_device, remove_background, resize_foreground

# Singleton Model Instance
_model_instance = None
_rembg_session = None

def get_model():
    global _model_instance, _rembg_session
    if _model_instance is None:
        print("[SF3D] Loading model into RAM (Singleton)...")
        device = get_device()
        if not (torch.cuda.is_available() or torch.backends.mps.is_available()):
            device = "cpu"
            
        print(f"[SF3D] Using device: {device}")
        
        _model_instance = SF3D.from_pretrained(
            "stabilityai/stable-fast-3d",
            config_name="config.yaml",
            weight_name="model.safetensors",
        )
        
        # [CRITICAL OPTIMIZATION FOR 4GB VRAM]
        # We rely strictly on torch.autocast for fp16 inference because calling .half() 
        # breaks the ViT/DINO backbone and produces bloated blobs!
        # _model_instance.half() # DO NOT DO THIS!
        _model_instance.to(device)
        _model_instance.eval()
        
        # Optimize CUDA
        if "cuda" in device:
            torch.backends.cudnn.benchmark = True
            
        _rembg_session = rembg.new_session()
        print("[SF3D] Model loaded successfully in FP16.")
    return _model_instance, _rembg_session, get_device()

def generate(img_path, output_dir, mode="full"):
    print(f"[SF3D] Starting generation for {img_path} in {mode} mode")
    model, rembg_session, device = get_model()
    
    # Preprocess Image
    print("[SF3D] Preprocessing image...")
    image = Image.open(img_path).convert("RGBA")
    
    image = remove_background(image, rembg_session)
    image = resize_foreground(image, 0.85)
    
    # Lite Mode config
    bake_resolution = 1024 if mode == "lite" else 2048
    remesh_option = "triangle"
    vertex_count = 30000 if mode == "lite" else -1
    
    print(f"[SF3D] Running inference (Bake: {bake_resolution}, Vertices: {vertex_count})...")
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    with torch.no_grad():
        with torch.autocast(
            device_type=device, dtype=torch.float16
        ) if "cuda" in device else nullcontext():
            # Ensure input is batched as [image]
            mesh, glob_dict = model.run_image(
                [image],
                bake_resolution=bake_resolution,
                remesh=remesh_option,
                vertex_count=vertex_count,
            )
            
    if torch.cuda.is_available():
        peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
        print(f"[SF3D] Peak Memory: {peak_mem:.2f} GB")
        
    # The run_image method returns a list of meshes if input is a list
    if isinstance(mesh, list):
        mesh = mesh[0]
        
    # Process Mesh
    from mesh_cleaner import clean_mesh
    print("[SF3D] Cleaning mesh...")
    mesh = clean_mesh(mesh)
    
    # Export Mesh
    from exporter import export_mesh
    print("[SF3D] Exporting mesh...")
    base_name = os.path.basename(img_path).replace("_conditioning.png", "").replace(".png", "")
    export_mesh(mesh, output_dir, base_name)
    
    print("[SF3D] Generation COMPLETE.")
    return True
