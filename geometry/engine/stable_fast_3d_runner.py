import os
import sys
import torch
import time
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from base_engine import Image3DEngine
from mesh_refiner import MeshRefiner, check_quality_gate
from mesh_cleaner import MeshCleaner
from mesh_validator import MeshValidator
from preview_renderer import PreviewRenderer
from report_exporter import ReportExporter

SF3D_DIR = "/Users/nguyenletruong/3d/stable-fast-3d"
sys.path.insert(0, SF3D_DIR)

try:
    from sf3d.system import SF3D
except ImportError:
    pass

class StableFast3DRunner(Image3DEngine):
    def __init__(self, device="cpu", precision=torch.float32):
        # MPS trên M2 bị treo khi chạy SF3D → dùng CPU cho ổn định
        os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
        self.device = torch.device("cpu")
        self.precision = precision
        self.model = None

    def load_model(self):
        if self.model is None:
            print("[StableFast3DRunner] Loading SF3D Model...")
            t0 = time.time()
            self.model = SF3D.from_pretrained(
                "stabilityai/stable-fast-3d",
                config_name="config.yaml",
                weight_name="model.safetensors",
            )
            
            self.model.to(self.device)
            self.model.eval()
            print(f"[StableFast3DRunner] Model loaded in {time.time() - t0:.1f}s")
            
            os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

    def generate3D(self, input_path: str, output_dir: str, npz_path: str = None, quality_json_path: str = None) -> dict:
        self.load_model()
        
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.basename(input_path).split('_')[0] + "_" + os.path.basename(input_path).split('_')[1] if '_' in os.path.basename(input_path) else "mesh"
        
        obj_path = os.path.join(output_dir, f"{base_name}_coarse_mesh.obj")
        glb_path = os.path.join(output_dir, f"{base_name}_coarse_mesh.glb")
        
        print("[StableFast3DRunner] Running Inference...")
        t0 = time.time()
        
        try:
            img = Image.open(input_path).convert("RGBA")
            
            with torch.no_grad():
                meshes, _ = self.model.run_image(
                    img,
                    bake_resolution=1024,
                    remesh="triangle",
                    vertex_count=50000,
                )
                    
            print("[StableFast3DRunner] Extracting Vertices and Faces...")
            meshes.export(obj_path)
            meshes.export(glb_path)
            
            inference_time = time.time() - t0
            print(f"[StableFast3DRunner] Mesh generated in {inference_time:.1f}s")
            
            if self.device.type == "mps":
                torch.mps.empty_cache()
            
            # Stage 4 → Stage 5 connection: Refine mesh using conditioning data
            refine_report = None
            if npz_path and os.path.exists(npz_path):
                print("[StableFast3DRunner] Refining mesh with conditioning data...")
                refiner = MeshRefiner(npz_path, quality_json_path)
                refine_report = refiner.refine(glb_path)
                # Also refine OBJ
                refiner_obj = MeshRefiner(npz_path)
                refiner_obj.refine(obj_path)
                print("[StableFast3DRunner] Mesh refinement complete.")
            else:
                print("[StableFast3DRunner] No conditioning data provided, skipping refinement.")
                
            cleaner = MeshCleaner(obj_path)
            cleaner.clean()
            
            validator = MeshValidator(obj_path)
            val_results = validator.validate()
            
            renderer = PreviewRenderer(obj_path, output_dir)
            preview_path = renderer.render()
            
            exporter = ReportExporter(val_results, inference_time, output_dir)
            report_path = exporter.export()
            
            return {
                "status": "success",
                "obj_path": obj_path,
                "glb_path": glb_path,
                "preview_path": preview_path,
                "report_path": report_path,
                "refine_report": refine_report
            }
            
        except Exception as e:
            print(f"[StableFast3DRunner] Error: {str(e)}")
            return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    
    runner = StableFast3DRunner()
    result = runner.generate3D(args.input, args.output)
    print("Result:", result)
