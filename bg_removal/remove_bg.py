import os
import sys
import time
import argparse
import urllib.request
import torch
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

# SAM2 imports
try:
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
except ImportError:
    print("Please install SAM-2: pip install git+https://github.com/facebookresearch/sam2.git")
    sys.exit(1)

def download_sam2_checkpoint(checkpoint_path):
    if not os.path.exists(checkpoint_path):
        print(f"Downloading SAM2 checkpoint to {checkpoint_path}...")
        url = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
        urllib.request.urlretrieve(url, checkpoint_path)
        print("Download complete.")

def process_image(image_path, output_dir):
    start_time = time.time()
    
    # 1. Device Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 2. Load and optimize image size
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    print(f"Original image size: {orig_w}x{orig_h}")
    
    max_dim = 2048
    if max(orig_w, orig_h) > max_dim:
        scale = max_dim / max(orig_w, orig_h)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)
        img = img.resize((new_w, new_h), Image.LANCZOS)
    else:
        new_w, new_h = orig_w, orig_h
        
    print(f"Processed image size: {new_w}x{new_h}")
    
    # 3. Load BiRefNet
    print("Loading BiRefNet...")
    birefnet = AutoModelForImageSegmentation.from_pretrained('ZhengPeng7/BiRefNet', trust_remote_code=True)
    birefnet.to(device)
    birefnet.to(torch.float32) # Fix MPS Half precision error
    birefnet.eval()
    
    # BiRefNet preprocessing
    transform_image = transforms.Compose([
        transforms.Resize((1024, 1024)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_images = transform_image(img).unsqueeze(0).to(device, dtype=torch.float32)
    
    print("Running BiRefNet...")
    with torch.no_grad():
        preds = birefnet(input_images)[-1].sigmoid().cpu()
    pred = preds[0].squeeze()
    
    pred_pil = transforms.ToPILImage()(pred)
    mask_biref = pred_pil.resize((new_w, new_h), Image.LANCZOS)
    mask_biref_arr = np.array(mask_biref)
    
    # 4. Load SAM2
    print("Loading SAM2...")
    checkpoint_path = os.path.join(os.path.dirname(__file__), "sam2_hiera_small.pt")
    download_sam2_checkpoint(checkpoint_path)
    
    sam2_model = build_sam2("sam2_hiera_s.yaml", checkpoint_path, device=device)
    predictor = SAM2ImagePredictor(sam2_model)
    
    print("Running SAM2...")
    img_arr = np.array(img)
    predictor.set_image(img_arr)
    
    # Dùng mask của BiRefNet để lấy Bounding Box cho SAM2
    y_indices, x_indices = np.where(mask_biref_arr > 128)
    if len(y_indices) > 0 and len(x_indices) > 0:
        x_min, x_max = np.min(x_indices), np.max(x_indices)
        y_min, y_max = np.min(y_indices), np.max(y_indices)
        
        pad = 20
        x_min, x_max = max(0, x_min - pad), min(new_w, x_max + pad)
        y_min, y_max = max(0, y_min - pad), min(new_h, y_max + pad)
        
        input_box = np.array([x_min, y_min, x_max, y_max])
        
        masks, scores, logits = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )
        mask_sam2_arr = (masks[0] * 255).astype(np.uint8)
    else:
        mask_sam2_arr = np.zeros_like(mask_biref_arr)
        
    # 5. Fusion and Post-processing
    print("Fusing masks and post-processing...")
    mask_final = np.maximum(mask_biref_arr, mask_sam2_arr)
    
    # Làm mượt (Smooth) và Feather viền 1-3px
    kernel = np.ones((3, 3), np.uint8)
    mask_final = cv2.morphologyEx(mask_final, cv2.MORPH_CLOSE, kernel)
    mask_final = cv2.GaussianBlur(mask_final, (5, 5), 0)
    
    # 6. Generate Outputs
    os.makedirs(output_dir, exist_ok=True)
    
    alpha_img = Image.fromarray(mask_final, mode='L')
    rgba_img = img.copy()
    rgba_img.putalpha(alpha_img)
    
    gray_bg = Image.new("RGBA", rgba_img.size, (128, 128, 128, 255))
    preview_img = Image.alpha_composite(gray_bg, rgba_img).convert("RGB")
    
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    rgba_path = os.path.join(output_dir, f"{base_name}_alpha.png")
    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    preview_path = os.path.join(output_dir, f"{base_name}_preview_gray.png")
    
    rgba_img.save(rgba_path)
    alpha_img.save(mask_path)
    preview_img.save(preview_path)
    
    end_time = time.time()
    
    print("=" * 40)
    print("✅ PROCESSING COMPLETE")
    print(f"Input: {image_path} ({orig_w}x{orig_h})")
    print(f"Output size: {new_w}x{new_h}")
    print(f"Saved: {output_dir}")
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print(f"Device used: {device}")
    print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="High-Quality BG Removal using BiRefNet + SAM2")
    parser.add_argument("--input", type=str, required=True, help="Input image path")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    args = parser.parse_args()
    
    process_image(args.input, args.output)
