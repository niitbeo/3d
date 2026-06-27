import cv2
import numpy as np
import argparse
import time
import os

def process_image(input_path, output_path, canvas_size=2048, target_scale=0.85):
    start_time = time.time()
    
    # 1. Load image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read {input_path}")
        
    orig_h, orig_w = img.shape[:2]
    
    if img.shape[2] != 4:
        raise ValueError("Input image must have an alpha channel (RGBA)")
        
    rgb = img[:, :, :3]
    alpha = img[:, :, 3]
    
    # 2. Alpha Validation & Shadow Removal
    # Threshold hard to remove faint shadows
    _, alpha = cv2.threshold(alpha, 15, 255, cv2.THRESH_TOZERO)
    
    # Fill small holes
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    alpha = cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    # 3. Bounding Box
    coords = cv2.findNonZero(alpha)
    if coords is None:
        raise ValueError("Image is completely transparent after alpha validation")
        
    x, y, w, h = cv2.boundingRect(coords)
    
    # Crop to Bounding Box
    rgb_cropped = rgb[y:y+h, x:x+w]
    alpha_cropped = alpha[y:y+h, x:x+w]
    
    # 4. Color Correction (CLAHE on L channel)
    lab = cv2.cvtColor(rgb_cropped, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    # Apply CLAHE very lightly
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    l_channel = clahe.apply(l_channel)
    lab_corrected = cv2.merge((l_channel, a_channel, b_channel))
    rgb_corrected = cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)
    
    # 5. Edge Refinement (Defringing / Matting)
    # Erode the alpha stronger to remove white halos from background extraction
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    alpha_eroded = cv2.erode(alpha_cropped, erode_kernel, iterations=2)
    # Blur slightly for smooth edges
    alpha_refined = cv2.GaussianBlur(alpha_eroded, (5, 5), 0)
    
    # Combine back to RGBA
    rgba_cropped = np.dstack((rgb_corrected, alpha_refined))
    
    # 6. Scale Normalization
    target_h = int(canvas_size * target_scale)
    scale_factor = target_h / h
    target_w = int(w * scale_factor)
    
    # Nếu ảnh ngang (width lớn) khiến target_w tràn ra khỏi canvas (ví dụ > 85% chiều rộng)
    if target_w > int(canvas_size * target_scale):
        target_w = int(canvas_size * target_scale)
        scale_factor = target_w / w
        target_h = int(h * scale_factor)
    
    # Resize the subject using INTER_AREA or LINEAR to avoid ringing and black halos on RGBA
    # To completely avoid halos, we separate RGB and Alpha, 
    # dilate pure foreground RGB into the semi-transparent/transparent areas, resize, then apply resized alpha.
    inpaint_mask = (alpha_refined < 255).astype(np.uint8)
    rgb_dilated = cv2.inpaint(rgb_corrected, inpaint_mask, 7, cv2.INPAINT_TELEA)
    
    rgb_resized = cv2.resize(rgb_dilated, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    alpha_resized = cv2.resize(alpha_refined, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    
    subject_resized = np.dstack((rgb_resized, alpha_resized))
    
    # 7. Center Alignment & Canvas Generation
    canvas = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    
    # Calculate placement coordinates
    start_y = (canvas_size - target_h) // 2
    start_x = (canvas_size - target_w) // 2
    
    canvas[start_y:start_y+target_h, start_x:start_x+target_w] = subject_resized
    
    # 8. Export Preview
    preview = np.full((canvas_size, canvas_size, 3), (128, 128, 128), dtype=np.uint8)
    alpha_norm = canvas[:, :, 3] / 255.0
    for c in range(3):
        preview[:, :, c] = (canvas[:, :, c] * alpha_norm + preview[:, :, c] * (1 - alpha_norm)).astype(np.uint8)
        
    # Save outputs
    cv2.imwrite(output_path, canvas)
    
    base_name, _ = os.path.splitext(output_path)
    preview_path = f"{base_name}_preview.png"
    cv2.imwrite(preview_path, preview)
    
    end_time = time.time()
    
    print("=" * 40)
    print("✅ CANONICALIZATION COMPLETE")
    print(f"Original Size:   {orig_w}x{orig_h}")
    print(f"Crop Size:       {w}x{h}")
    print(f"Scale Factor:    {scale_factor:.3f}x")
    print(f"Final Canvas:    {canvas_size}x{canvas_size}")
    print(f"Padding height:  {(1 - target_scale)/2 * 100:.1f}% top/bottom")
    print(f"Execution Time:  {end_time - start_time:.3f}s")
    print("=" * 40)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input RGBA alpha image")
    parser.add_argument("--output", required=True, help="Output canonical image path")
    parser.add_argument("--size", type=int, default=2048, help="Canvas size")
    parser.add_argument("--scale", type=float, default=0.85, help="Subject height scale")
    
    args = parser.parse_args()
    
    process_image(args.input, args.output, args.size, args.scale)
