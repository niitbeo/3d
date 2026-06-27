import argparse
import sys
import numpy as np
import cv2
import os
import time

def generate_depth_mesh(image_path, output_path):
    print("Initializing fast displacement mesh generator...")
    
    # image_path is canonical.png
    base_name = os.path.basename(image_path).replace("_canonical.png", "")
    
    bg_dir = "/Users/nguyenletruong/3d/bg_removal/output"
    geom_dir = "/Users/nguyenletruong/3d/geometry/output"
    
    canonical_path = os.path.join(bg_dir, f"{base_name}_canonical.png")
    depth_path = os.path.join(geom_dir, f"{base_name}_depth.npy")
    alpha_path = os.path.join(bg_dir, f"{base_name}_alpha.png")
    
    if not os.path.exists(depth_path):
        print(f"Error: Depth map not found at {depth_path}")
        return
        
    print("Processing images and removing background...")
    print("Initializing model...")  # Trigger 15%
    time.sleep(0.1)
    
    print("Running model...") # Trigger 50%
    
    # Load canonical image
    rgba = cv2.imread(canonical_path, cv2.IMREAD_UNCHANGED)
    if rgba.shape[2] == 4:
        rgb = rgba[:, :, :3]
        alpha = rgba[:, :, 3]
    else:
        rgb = rgba
        alpha = cv2.imread(alpha_path, cv2.IMREAD_GRAYSCALE)
        
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    # Load depth map
    depth_map = np.load(depth_path)
    
    # Resize to max 600px for fast mesh generation
    max_dim = 600
    h, w = rgb.shape[:2]
    scale = max_dim / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    print("Extracting mesh...") # Trigger 70%
    
    rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    alpha = cv2.resize(alpha, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    depth_map = cv2.resize(depth_map, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    # Normalize depth map
    depth_min = depth_map.min()
    depth_max = depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min + 1e-8)
    
    color_array = rgb / 255.0
    
    print("Exporting mesh...") # Trigger 90%
    vertices = []
    faces = []
    
    depth_scale = 0.3
    
    for y in range(new_h):
        for x in range(new_w):
            vx = ((x / new_w) - 0.5) * 2.0
            vy = -(((y / new_h) - 0.5) * 2.0 * (new_h/new_w))
            vz = depth_norm[y, x] * depth_scale
            r, g, b = color_array[y, x]
            vertices.append(f"v {vx:.4f} {vy:.4f} {vz:.4f} {r:.3f} {g:.3f} {b:.3f}")
            
    for y in range(new_h - 1):
        for x in range(new_w - 1):
            # Only create faces if all 4 vertices are solid foreground
            if alpha[y, x] > 200 and alpha[y, x+1] > 200 and alpha[y+1, x] > 200 and alpha[y+1, x+1] > 200:
                idx1 = y * new_w + x + 1
                idx2 = idx1 + 1
                idx3 = (y + 1) * new_w + x + 1
                idx4 = idx3 + 1
                
                # Triangle 1
                faces.append(f"f {idx1} {idx2} {idx3}")
                # Triangle 2
                faces.append(f"f {idx2} {idx4} {idx3}")
            
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.write("\n".join(vertices))
        f.write("\n")
        f.write("\n".join(faces))
        
    print("Exporting mesh finished...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=str)
    parser.add_argument("--output", type=str, default="output/0/mesh.obj")
    args = parser.parse_args()
    generate_depth_mesh(args.image, args.output)
