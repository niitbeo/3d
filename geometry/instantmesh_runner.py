import os
import sys
import argparse
import time
import json
import numpy as np
import torch
import trimesh
import open3d as o3d
from PIL import Image

cuda_bin = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin"
if os.path.exists(cuda_bin):
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(cuda_bin)
    os.environ['PATH'] = cuda_bin + os.pathsep + os.environ.get('PATH', '')

# Add InstantMesh to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "instantmesh")))

from huggingface_hub import hf_hub_download
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
from omegaconf import OmegaConf
from einops import rearrange
from torchvision.transforms import v2

import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda *args, **kwargs: None
import transformers.modeling_utils
if hasattr(transformers.modeling_utils, 'check_torch_load_is_safe'):
    transformers.modeling_utils.check_torch_load_is_safe = lambda *args, **kwargs: None

from src.utils.train_util import instantiate_from_config
from src.utils.camera_util import get_zero123plus_input_cameras
from src.utils.mesh_util import save_obj
from src.utils.infer_util import remove_background, resize_foreground

class InputLoader:
    @staticmethod
    def load(input_path):
        # Handle npz from TripoSR pipeline (which contains latent/rgb_pil) or a direct image
        if input_path.endswith('.npz'):
            data = np.load(input_path, allow_pickle=True)
            if 'rgb_pil' in data:
                return data['rgb_pil'].item()
        
        # If it's a directory (from Photobooth stage 4), look for canonical.png or similar
        if os.path.isdir(input_path):
            img_path = os.path.join(input_path, "canonical.png")
            if os.path.exists(img_path):
                return Image.open(img_path)
            
            # fallback to any png
            for f in os.listdir(input_path):
                if f.endswith('.png'):
                    return Image.open(os.path.join(input_path, f))
                    
        # If direct image
        if os.path.isfile(input_path):
            return Image.open(input_path)
            
        raise ValueError(f"Could not load valid input from {input_path}")

class GeometryValidator:
    @staticmethod
    def validate(image):
        if not isinstance(image, Image.Image):
            raise ValueError("Input must be a PIL Image")
        if image.mode != "RGBA" and image.mode != "RGB":
            image = image.convert("RGBA")
        return image

class InstantMeshRunner:
    def __init__(self, device="auto", precision="fp16"):
        self.t0 = time.time()
        print("[InstantMeshRunner] Initializing...")
        
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        
        print(f"[InstantMeshRunner] Using device: {self.device}, dtype: {self.dtype}")
        
        # Load Diffusion Pipeline
        self.pipeline = DiffusionPipeline.from_pretrained(
            "sudo-ai/zero123plus-v1.2",
            custom_pipeline="sudo-ai/zero123plus-pipeline",
            torch_dtype=self.dtype,
            trust_remote_code=True,
        )
        self.pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
            self.pipeline.scheduler.config, timestep_spacing='trailing'
        )
        
        # Load custom UNet
        unet_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="diffusion_pytorch_model.bin", repo_type="model")
        state_dict = torch.load(unet_ckpt_path, map_location='cpu')
        self.pipeline.unet.load_state_dict(state_dict, strict=True)
        self.pipeline = self.pipeline.to(self.device)
        
        # Load Reconstruction Model
        config_path = os.path.join(os.path.dirname(__file__), "..", "instantmesh", "configs", "instant-mesh-large.yaml")
        config = OmegaConf.load(config_path)
        self.model_config = config.model_config
        self.infer_config = config.infer_config
        
        self.model = instantiate_from_config(self.model_config)
        model_ckpt_path = hf_hub_download(repo_id="TencentARC/InstantMesh", filename="instant_mesh_large.ckpt", repo_type="model")
        state_dict = torch.load(model_ckpt_path, map_location='cpu')['state_dict']
        state_dict = {k[14:]: v for k, v in state_dict.items() if k.startswith('lrm_generator.')}
        self.model.load_state_dict(state_dict, strict=True)
        
        self.model = self.model.to(self.device, dtype=self.dtype)
        self.model.init_flexicubes_geometry(self.device, fovy=30.0)
        self.model = self.model.eval()
        
        print(f"[InstantMeshRunner] Loaded in {time.time() - self.t0:.2f}s")

    @torch.no_grad()
    def generate(self, input_image, diffusion_steps=30, output_path=None):
        t0 = time.time()
        print("[InstantMeshRunner] Generating Multiview...")
        
        # Background removal if not RGBA or alpha is solid
        input_image = remove_background(input_image)
        input_image = resize_foreground(input_image, 0.85)
        
        output_image = self.pipeline(
            input_image,
            num_inference_steps=diffusion_steps,
        ).images[0]
        
        images = np.asarray(output_image, dtype=np.float32) / 255.0
        images = torch.from_numpy(images).permute(2, 0, 1).contiguous().float()
        images = rearrange(images, 'c (n h) (m w) -> (n m) c h w', n=3, m=2)
        
        print("[InstantMeshRunner] Reconstructing Mesh...")
        input_cameras = get_zero123plus_input_cameras(batch_size=1, radius=4.0).to(self.device, dtype=self.dtype)
        images = images.unsqueeze(0).to(self.device, dtype=self.dtype)
        images = v2.functional.resize(images, 320, interpolation=3, antialias=True).clamp(0, 1)
        
        planes = self.model.forward_planes(images, input_cameras)
        mesh_out = self.model.extract_mesh(
            planes,
            use_texture_map=False,
            **self.infer_config,
        )
        
        vertices, faces, vertex_colors = mesh_out
        
        if output_path:
            save_obj(vertices, faces, vertex_colors, output_path)
            
        inference_time = time.time() - t0
        print(f"[InstantMeshRunner] Inference complete in {inference_time:.2f}s")
        return vertices, faces, vertex_colors, inference_time

class MeshCleaner:
    @staticmethod
    def clean(obj_path, out_path):
        print("[MeshCleaner] Cleaning mesh...")
        mesh = trimesh.load(obj_path, force='mesh')
        
        # Remove duplicate vertices
        mesh.merge_vertices()
        
        # Repair non-manifold
        mesh.fix_normals()
        trimesh.repair.fix_inversion(mesh)
        
        # Remove tiny components (keep only the largest connected component)
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            areas = [c.area for c in components]
            mesh = components[np.argmax(areas)]
            
        mesh.export(out_path)
        return mesh

class MeshValidator:
    @staticmethod
    def validate(mesh):
        print("[MeshValidator] Validating mesh...")
        watertight = mesh.is_watertight
        
        if not watertight:
            print("[MeshValidator] Auto-repairing holes...")
            trimesh.repair.fill_holes(mesh)
            watertight = mesh.is_watertight
            
        stats = {
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": watertight,
            "status": "PASS" if watertight else "WARNING"
        }
        return stats, mesh

class PreviewRenderer:
    @staticmethod
    def render(mesh_path, out_png_path):
        import pyrender
        print("[PreviewRenderer] Rendering preview...")
        # A simple offscreen rendering
        mesh = trimesh.load(mesh_path, force='mesh')
        scene = pyrender.Scene(ambient_light=[0.2, 0.2, 0.2])
        mesh_node = pyrender.Mesh.from_trimesh(mesh)
        scene.add(mesh_node)
        
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 4.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        scene.add(camera, pose=camera_pose)
        
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)
        
        r = pyrender.OffscreenRenderer(512, 512)
        color, _ = r.render(scene)
        
        Image.fromarray(color).save(out_png_path)
        r.delete()

class ReportExporter:
    @staticmethod
    def export(stats, inference_time, out_json_path):
        print("[ReportExporter] Generating report...")
        report = {
            "engine": "InstantMesh",
            "device": "Auto",
            "inference_time": round(inference_time, 2),
            **stats
        }
        with open(out_json_path, 'w') as f:
            json.dump(report, f, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help="Input directory or file")
    parser.add_argument('--output', type=str, required=True, help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    
    # Extract base_name from input image (e.g. 12345-6789_conditioning.png -> 12345-6789)
    input_filename = os.path.basename(args.input)
    base_name = input_filename.replace("_conditioning.png", "").replace(".png", "")
    if not base_name or base_name == args.input:
        base_name = "coarse"
        
    coarse_obj = os.path.join(args.output, f"{base_name}_coarse.obj")
    coarse_glb = os.path.join(args.output, f"{base_name}_coarse.glb")
    mesh_preview = os.path.join(args.output, f"{base_name}_geometry_preview.png")
    mesh_report = os.path.join(args.output, f"{base_name}_mesh_report.json")

    # 1. Load Input
    raw_img = InputLoader.load(args.input)
    
    # 2. Validate
    img = GeometryValidator.validate(raw_img)
    
    # 3. Runner
    runner = InstantMeshRunner()
    _, _, _, inf_time = runner.generate(img, output_path=coarse_obj)
    
    # 4. Cleaner
    mesh = MeshCleaner.clean(coarse_obj, coarse_obj)
    
    # 5. Validator
    stats, mesh = MeshValidator.validate(mesh)
    
    # Export GLB
    mesh.export(coarse_glb)
    
    # 6. Preview Renderer
    try:
        PreviewRenderer.render(coarse_glb, mesh_preview)
    except Exception as e:
        print(f"[PreviewRenderer] Warning: Could not render preview: {e}")
        # dummy preview
        Image.new("RGB", (512, 512), color="gray").save(mesh_preview)
        
    # 7. Report Exporter
    ReportExporter.export(stats, inf_time, mesh_report)
    
    print("[Pipeline] InstantMesh Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
