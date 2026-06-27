import os
import time
import argparse
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import pipeline

class DepthModel:
    def __init__(self, device):
        self.device = device
        print("Loading Depth Anything V2 (Large)...")
        self.pipe = pipeline(task="depth-estimation", model="depth-anything/Depth-Anything-V2-Large-hf", device=self.device)
        
    def predict(self, rgb_image):
        result = self.pipe(rgb_image)
        predicted_depth = result["predicted_depth"]
        depth_map = predicted_depth.squeeze().cpu().numpy()
        return depth_map

class DepthPostProcess:
    @staticmethod
    def process(depth_map, mask):
        h, w = mask.shape
        if depth_map.shape != (h, w):
            depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_CUBIC)
            
        depth_smoothed = cv2.bilateralFilter(depth_map, d=5, sigmaColor=0.1, sigmaSpace=5)
        
        # Inpaint foreground depth outward into the background to prevent edge gradients
        depth_smoothed = cv2.inpaint(depth_smoothed, (mask == 0).astype(np.uint8), 5, cv2.INPAINT_TELEA)
        
        # Now apply alpha blending with min_depth
        min_depth = np.min(depth_map)
        alpha_norm = mask.astype(np.float32) / 255.0
        depth_final = depth_smoothed * alpha_norm + min_depth * (1.0 - alpha_norm)
        
        return depth_final

class NormalGenerator:
    @staticmethod
    def generate(depth_map, mask):
        depth_norm = (depth_map - np.min(depth_map)) / (np.max(depth_map) - np.min(depth_map) + 1e-8)
        
        # Để tránh Sobel tạo viền đỏ gắt ở biên alpha, ta cần inpaint depth_map ra ngoài 
        # trước khi tính gradient. Do đầu vào depth_map đã blend với min_depth ở background,
        # ta cần lấy lại vùng foreground và inpaint ra.
        fg_mask = (mask > 0).astype(np.uint8)
        depth_inpainted = cv2.inpaint(depth_norm.astype(np.float32), 1 - fg_mask, 5, cv2.INPAINT_TELEA)
        
        dzdx = cv2.Sobel(depth_inpainted, cv2.CV_32F, 1, 0, ksize=3)
        dzdy = cv2.Sobel(depth_inpainted, cv2.CV_32F, 0, 1, ksize=3)
        
        normal_x = -dzdx
        normal_y = dzdy
        normal_z = np.ones_like(depth_inpainted)
        
        normals = np.dstack((normal_x, normal_y, normal_z))
        
        norm = np.linalg.norm(normals, axis=2, keepdims=True) + 1e-8
        normals = normals / norm
        
        normals_rgb = ((normals + 1.0) / 2.0 * 255.0).astype(np.float32)
        alpha_norm_3 = (mask.astype(np.float32) / 255.0)[..., np.newaxis]
        bg_normal = np.array([128.0, 128.0, 255.0], dtype=np.float32)
        normals_blended = normals_rgb * alpha_norm_3 + bg_normal * (1.0 - alpha_norm_3)
        
        return normals_blended.astype(np.uint8)

class GeometryExporter:
    @staticmethod
    def export(output_dir, base_name, rgb, depth, normal):
        os.makedirs(output_dir, exist_ok=True)
        
        depth_npy_path = os.path.join(output_dir, f"{base_name}_depth.npy")
        depth_png_path = os.path.join(output_dir, f"{base_name}_depth.png")
        normal_path = os.path.join(output_dir, f"{base_name}_normal.png")
        preview_path = os.path.join(output_dir, f"{base_name}_geometry_preview.png")
        
        np.save(depth_npy_path, depth)
        
        depth_min, depth_max = np.min(depth), np.max(depth)
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth_16 = (depth_norm * 65535).astype(np.uint16)
        cv2.imwrite(depth_png_path, depth_16)
        
        cv2.imwrite(normal_path, cv2.cvtColor(normal, cv2.COLOR_RGB2BGR))
        
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        depth_preview_8 = (depth_norm * 255).astype(np.uint8)
        depth_preview_bgr = cv2.cvtColor(depth_preview_8, cv2.COLOR_GRAY2BGR)
        normal_bgr = cv2.cvtColor(normal, cv2.COLOR_RGB2BGR)
        
        preview = np.hstack((bgr, depth_preview_bgr, normal_bgr))
        if preview.shape[1] > 3072:
            scale = 3072 / preview.shape[1]
            preview = cv2.resize(preview, (3072, int(preview.shape[0]*scale)))
            
        cv2.imwrite(preview_path, preview)

def process_geometry(input_path, output_dir):
    start_time = time.time()
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    img_rgba = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img_rgba is None or img_rgba.shape[2] != 4:
        raise ValueError("Input must be a valid RGBA image")
        
    h, w = img_rgba.shape[:2]
    print(f"Original Resolution: {w}x{h}")
    
    alpha = img_rgba[:, :, 3]
    rgb = img_rgba[:, :, :3]
    
    alpha_norm = alpha / 255.0
    gray_bg = np.full_like(rgb, 128)
    for c in range(3):
        rgb[:, :, c] = (rgb[:, :, c] * alpha_norm + gray_bg[:, :, c] * (1 - alpha_norm)).astype(np.uint8)
        
    rgb_pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))
    
    model = DepthModel(device)
    raw_depth = model.predict(rgb_pil)
    
    if device.type == 'mps':
        print(f"Peak Memory: {torch.mps.current_allocated_memory() / 1024**2:.1f} MB")
    elif device.type == 'cuda':
        print(f"Peak Memory: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB")
        
    print(f"Depth Resolution (Raw): {raw_depth.shape[1]}x{raw_depth.shape[0]}")
    
    processed_depth = DepthPostProcess.process(raw_depth, alpha)
    normals = NormalGenerator.generate(processed_depth, alpha)
    
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    base_name = base_name.replace('_canonical', '').replace('_alpha', '')
    
    rgb_original = img_rgba[:, :, :3]
    rgb_original = cv2.cvtColor(rgb_original, cv2.COLOR_BGR2RGB)
    
    GeometryExporter.export(output_dir, base_name, rgb_original, processed_depth, normals)
    
    print("=" * 40)
    print("GEOMETRY ESTIMATION COMPLETE")
    print(f"Inference Time: {time.time() - start_time:.2f}s")
    print(f"Output Directory: {output_dir}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input Canonical RGBA Image")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    process_geometry(args.input, args.output)
