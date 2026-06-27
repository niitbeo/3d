import os
import argparse
import time
import json
import numpy as np
import cv2

class GeometryValidator:
    @staticmethod
    def validate(rgb, depth, normal, alpha):
        if rgb.shape[:2] != depth.shape or rgb.shape[:2] != normal.shape[:2]:
            raise ValueError(f"Shape mismatch: RGB {rgb.shape[:2]}, Depth {depth.shape}, Normal {normal.shape[:2]}")
        
        # Check alpha empty
        if np.sum(alpha > 128) == 0:
            raise ValueError("Alpha mask is completely empty.")
            
        print("[GeometryValidator] Resolution and structural validation passed.")
        return True

class FeatureFusion:
    @staticmethod
    def fuse(rgb, depth, normal, alpha):
        # Simply keeping them aligned in memory. No resizing or normalization here.
        # This module ensures they are bundled together.
        return rgb, depth, normal, alpha

class OcclusionDetector:
    @staticmethod
    def detect(depth, normal, alpha):
        # Detect sharp discontinuities in depth
        depth_valid = depth * (alpha > 128)
        
        # Normalize depth for consistent gradient thresholding
        d_min, d_max = np.min(depth_valid[alpha>128]), np.max(depth_valid[alpha>128])
        if d_max - d_min > 0:
            d_norm = (depth_valid - d_min) / (d_max - d_min)
        else:
            d_norm = depth_valid
            
        sobelx = cv2.Sobel(d_norm, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(d_norm, cv2.CV_32F, 0, 1, ksize=3)
        depth_grad = np.sqrt(sobelx**2 + sobely**2)
        
        # Detect sharp discontinuities in Normal
        # Convert to float [-1, 1]
        n_float = (normal.astype(np.float32) / 255.0) * 2.0 - 1.0
        n_float = n_float * np.expand_dims(alpha > 128, axis=-1)
        
        nx_grad = cv2.Sobel(n_float[:,:,0], cv2.CV_32F, 1, 0, ksize=3)
        ny_grad = cv2.Sobel(n_float[:,:,1], cv2.CV_32F, 0, 1, ksize=3)
        nz_grad = cv2.Sobel(n_float[:,:,2], cv2.CV_32F, 1, 1, ksize=3)
        normal_grad = np.sqrt(nx_grad**2 + ny_grad**2 + nz_grad**2)
        
        # Combine gradients
        combined_grad = depth_grad * 0.7 + normal_grad * 0.3
        
        # Threshold for occlusion (sharp jumps)
        # Using a conservative threshold
        occlusion_map = (combined_grad > 0.5).astype(np.float32)
        
        # Clean up occlusion map
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        occlusion_map = cv2.dilate(occlusion_map, kernel, iterations=1)
        occlusion_map = cv2.GaussianBlur(occlusion_map, (5, 5), 0)
        
        return occlusion_map

class VisibilityEstimator:
    @staticmethod
    def estimate(normal, alpha):
        # Decode Normal map to Z component
        # Assuming OpenGL normal map where B channel is Z (or R channel depending on order)
        # BGR format in OpenCV: Z is typically the 0-th (B) or 2-nd (R) channel.
        # In our geometry.py, we mapped Z to Blue (index 0). Let's extract Z:
        # Actually normal.py in geometry.py: canvas[..., 0] = z_c (Blue = Z)
        z_channel = normal[:, :, 0].astype(np.float32) / 255.0
        z_vector = (z_channel - 0.5) * 2.0
        
        # Visibility probability is directly proportional to how much the surface faces the camera
        # Z = 1.0 -> Facing directly -> Visibility 1.0
        # Z = 0.0 -> Facing sideways (shoulder/edge) -> Visibility 0.0
        # Z < 0.0 -> Facing away -> Visibility 0.0
        
        visibility = np.clip(z_vector, 0.0, 1.0)
        
        # Smooth transition at the very edges of the mask
        alpha_f = alpha.astype(np.float32) / 255.0
        visibility = visibility * alpha_f
        
        # Further refine: very thin parts (like hair ends) have lower visibility
        eroded = cv2.erode(alpha, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        dist = cv2.distanceTransform(eroded, cv2.DIST_L2, 3)
        cv2.normalize(dist, dist, 0, 1.0, cv2.NORM_MINMAX)
        dist_factor = np.clip(dist * 2.0, 0.0, 1.0)
        
        visibility = visibility * (0.8 + 0.2 * dist_factor)
        visibility = np.clip(visibility, 0.0, 1.0)
        
        return visibility.astype(np.float32)

class ConfidenceEstimator:
    @staticmethod
    def estimate(alpha, depth, occlusion, visibility):
        # Base confidence from mask
        conf = (alpha.astype(np.float32) / 255.0)
        
        # Reduce confidence in occluded areas (AI needs to hallucinate behind occlusions)
        # occlusion is [0, 1]. High occlusion means low confidence.
        conf = conf * (1.0 - occlusion * 0.8)
        
        # Reduce confidence slightly on edges/shoulders (where visibility is low)
        conf = conf * (0.6 + 0.4 * visibility)
        
        # Smooth confidence map
        conf = cv2.GaussianBlur(conf, (15, 15), 0)
        
        return np.clip(conf, 0.0, 1.0).astype(np.float32)

class QualityEvaluator:
    @staticmethod
    def evaluate(depth, normal, alpha, visibility, confidence):
        alpha_mask = alpha > 128
        fg_pixels = np.sum(alpha_mask)
        fg_ratio = fg_pixels / (alpha.shape[0] * alpha.shape[1])
        
        if fg_pixels == 0:
            return {"status": "FAIL", "overall_score": 0.0}
            
        # 1. Depth Score (Smoothness & Range)
        depth_fg = depth[alpha_mask]
        depth_range = np.max(depth_fg) - np.min(depth_fg)
        depth_score = min(1.0, depth_range / 50.0) if depth_range > 0 else 0.0
        
        # 2. Normal Score (Detail variance)
        norm_var = np.var(normal[alpha_mask])
        normal_score = min(1.0, norm_var / 3000.0)
        
        # 3. Mask Score (Solidness)
        mask_score = 1.0 if fg_ratio > 0.05 and fg_ratio < 0.95 else 0.5
        
        # 4. Visibility Score (Is there enough front-facing area?)
        vis_mean = np.mean(visibility[alpha_mask])
        vis_score = min(1.0, vis_mean * 1.5)
        
        # 5. Confidence Score
        conf_mean = np.mean(confidence[alpha_mask])
        conf_score = conf_mean
        
        # 6. Geometry Score (Combined)
        geom_score = (depth_score * 0.4 + normal_score * 0.6)
        
        # Overall Score
        overall = (geom_score * 0.4 + vis_score * 0.3 + conf_score * 0.2 + mask_score * 0.1)
        
        status = "PASS" if overall >= 0.85 else "FAIL" # Set to 0.85 as a practical threshold
        
        return {
            "depth_score": round(float(depth_score), 3),
            "normal_score": round(float(normal_score), 3),
            "mask_score": round(float(mask_score), 3),
            "visibility_score": round(float(vis_score), 3),
            "confidence_score": round(float(conf_score), 3),
            "geometry_score": round(float(geom_score), 3),
            "overall_score": round(float(overall), 3),
            "foreground_ratio": round(float(fg_ratio), 3),
            "status": status
        }

class GeometryFeatureBuilder:
    @staticmethod
    def export_npz(output_dir, base_name, rgb, depth, normal, alpha, visibility, confidence):
        npz_path = os.path.join(output_dir, f"{base_name}_geometry_feature.npz")
        
        metadata = {
            "version": "1.0",
            "pipeline": "Production Geometry Conditioning",
            "resolution": list(rgb.shape[:2]),
            "bbox": cv2.boundingRect(alpha)
        }
        
        np.savez_compressed(
            npz_path,
            rgb=rgb,
            depth=depth,
            normal=normal,
            alpha=alpha,
            visibility=visibility,
            confidence=confidence,
            metadata=json.dumps(metadata)
        )
        return npz_path

class DebugExporter:
    @staticmethod
    def export(output_dir, base_name, rgb, depth, normal, visibility, confidence, occlusion, alpha):
        # Prepare components
        def prep(img, cmap=None):
            if img.dtype != np.uint8:
                img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            if cmap is not None:
                img = cv2.applyColorMap(img, cmap)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            img[alpha < 128] = 0
            return cv2.resize(img, (512, 512))

        r_viz = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGBA2BGR), (512, 512))
        r_viz[cv2.resize(alpha, (512, 512)) < 128] = 0
        
        d_viz = prep(depth, cv2.COLORMAP_INFERNO)
        n_viz = prep(normal)
        v_viz = prep(visibility, cv2.COLORMAP_PLASMA)
        c_viz = prep(confidence, cv2.COLORMAP_VIRIDIS)
        o_viz = prep(occlusion, cv2.COLORMAP_HOT)
        
        # Create 2x3 Grid
        row1 = np.hstack((r_viz, d_viz, n_viz))
        row2 = np.hstack((v_viz, c_viz, o_viz))
        grid = np.vstack((row1, row2))
        
        # Add labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        labels = ["RGB", "Depth", "Normal", "Visibility", "Confidence", "Occlusion"]
        for i in range(2):
            for j in range(3):
                idx = i * 3 + j
                cv2.putText(grid, labels[idx], (j * 512 + 20, i * 512 + 40), font, 1.2, (255, 255, 255), 2, cv2.LINE_AA)
        
        cond_path = os.path.join(output_dir, f"{base_name}_conditioning.png")
        cv2.imwrite(cond_path, grid)
        
        # Save visibility map
        vis_path = os.path.join(output_dir, f"{base_name}_visibility_map.png")
        cv2.imwrite(vis_path, prep(visibility, cv2.COLORMAP_PLASMA))
        
        # Save confidence map
        conf_path = os.path.join(output_dir, f"{base_name}_confidence_map.png")
        cv2.imwrite(conf_path, prep(confidence, cv2.COLORMAP_VIRIDIS))
        
        return cond_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--canonical", required=True, help="Path to canonical image")
    parser.add_argument("--depth", required=True, help="Path to depth.npy")
    parser.add_argument("--normal", required=True, help="Path to normal.png")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    
    start_time = time.time()
    
    base_name = os.path.splitext(os.path.basename(args.canonical))[0].replace("_canonical", "")
    os.makedirs(args.output, exist_ok=True)
    
    print("Loading inputs...")
    img = cv2.imread(args.canonical, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(img[:,:,:3], cv2.COLOR_BGR2RGB)
    alpha = img[:,:,3]
    
    depth = np.load(args.depth)
    normal = cv2.imread(args.normal, cv2.IMREAD_COLOR)
    
    # 1. Validation
    GeometryValidator.validate(rgb, depth, normal, alpha)
    
    # 2. Fusion
    rgb, depth, normal, alpha = FeatureFusion.fuse(rgb, depth, normal, alpha)
    
    # 3. Occlusion Detection
    print("Detecting occlusions...")
    occ = OcclusionDetector.detect(depth, normal, alpha)
    
    # 4. Visibility Estimation
    print("Estimating visibility probability...")
    vis = VisibilityEstimator.estimate(normal, alpha)
    np.save(os.path.join(args.output, f"{base_name}_visibility.npy"), vis)
    
    # 5. Confidence Estimation
    print("Estimating confidence...")
    conf = ConfidenceEstimator.estimate(alpha, depth, occ, vis)
    
    # 6. Quality Evaluation
    print("Evaluating production quality...")
    metrics = QualityEvaluator.evaluate(depth, normal, alpha, vis, conf)
    json_path = os.path.join(args.output, f"{base_name}_quality.json")
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
        
    # 7. Geometry Feature Export
    print("Building geometry features (.npz)...")
    GeometryFeatureBuilder.export_npz(args.output, base_name, rgb, depth, normal, alpha, vis, conf)
    
    # 8. Debug Export
    print("Generating conditioning debug grid...")
    DebugExporter.export(args.output, base_name, rgb, depth, normal, vis, conf, occ, alpha)
    
    end_time = time.time()
    
    print("=" * 40)
    print("PRODUCTION CONDITIONING COMPLETE")
    print(f"Base name:       {base_name}")
    print(f"Execution Time:  {end_time - start_time:.3f}s")
    print(f"Status:          {metrics['status']}")
    print(f"Overall Score:   {metrics['overall_score']}")
    print("=" * 40)

if __name__ == "__main__":
    main()
