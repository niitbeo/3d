"""
MeshRefiner — Post-process SF3D mesh using conditioning data from Stage 4.

Dùng dữ liệu conditioning (depth, normal, visibility, confidence) 
để cải thiện mesh sau khi SF3D tạo xong:
1. Confidence-based cleanup: xóa faces ở vùng confidence thấp
2. Normal smoothing: blend vertex normals với normal map ước lượng
3. Depth-guided vertex correction (conservative): điều chỉnh vertex depth nhẹ
4. Visibility-weighted vertex color softening
"""

import os
import json
import numpy as np
import trimesh
import cv2


class MeshRefiner:
    """Post-process SF3D mesh using geometry conditioning data."""

    def __init__(self, npz_path: str, quality_json_path: str = None):
        """
        Load geometry features from conditioning NPZ.
        
        Args:
            npz_path: Path to {name}_geometry_feature.npz from Stage 4
            quality_json_path: Optional path to {name}_quality.json
        """
        self.npz_path = npz_path
        self.quality_json_path = quality_json_path
        self.features = None
        self.quality = None
        self._load_features()

    def _load_features(self):
        """Load NPZ and quality data."""
        if not os.path.exists(self.npz_path):
            print(f"[MeshRefiner] WARNING: NPZ not found: {self.npz_path}")
            return

        print(f"[MeshRefiner] Loading conditioning data from {os.path.basename(self.npz_path)}")
        data = np.load(self.npz_path)
        self.features = {
            "rgb": data["rgb"],
            "depth": data["depth"],
            "normal": data["normal"],
            "alpha": data["alpha"],
            "visibility": data["visibility"],
            "confidence": data["confidence"],
        }

        if self.quality_json_path and os.path.exists(self.quality_json_path):
            with open(self.quality_json_path, 'r') as f:
                self.quality = json.load(f)
            print(f"[MeshRefiner] Quality score: {self.quality.get('overall_score', 'N/A')} ({self.quality.get('status', 'N/A')})")

    def refine(self, mesh_path: str, output_path: str = None) -> dict:
        """
        Apply refinements to mesh and return report.
        
        Args:
            mesh_path: Path to input mesh (.obj or .glb)
            output_path: Optional output path. If None, overwrites input.
            
        Returns:
            dict with refinement report
        """
        if self.features is None:
            print("[MeshRefiner] No conditioning data available, skipping refinement.")
            return {"status": "skipped", "reason": "no_conditioning_data"}

        if output_path is None:
            output_path = mesh_path

        print(f"[MeshRefiner] Loading mesh: {os.path.basename(mesh_path)}")
        
        try:
            scene_or_mesh = trimesh.load(mesh_path)
            
            # Handle Scene (GLB) vs single Trimesh
            if isinstance(scene_or_mesh, trimesh.Scene):
                meshes = [g for g in scene_or_mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
                if not meshes:
                    return {"status": "skipped", "reason": "no_trimesh_in_scene"}
                mesh = meshes[0]
                is_scene = True
            else:
                mesh = scene_or_mesh
                is_scene = False

            original_verts = len(mesh.vertices)
            original_faces = len(mesh.faces)

            report = {
                "original_vertices": original_verts,
                "original_faces": original_faces,
                "steps": []
            }

            # Step 1: Confidence-based cleanup
            removed = self._remove_low_confidence_faces(mesh)
            report["steps"].append({"name": "confidence_cleanup", "faces_removed": removed})

            # Step 2: Normal smoothing
            smoothed = self._smooth_normals(mesh)
            report["steps"].append({"name": "normal_smoothing", "applied": smoothed})

            # Step 3: Conservative depth correction
            corrected = self._correct_depth_conservative(mesh)
            report["steps"].append({"name": "depth_correction", "vertices_adjusted": corrected})

            report["final_vertices"] = len(mesh.vertices)
            report["final_faces"] = len(mesh.faces)
            report["status"] = "success"

            # Export refined mesh
            if is_scene:
                scene_or_mesh.export(output_path)
            else:
                mesh.export(output_path)

            print(f"[MeshRefiner] Refinement complete: {original_faces} → {len(mesh.faces)} faces")
            return report

        except Exception as e:
            print(f"[MeshRefiner] Error during refinement: {e}")
            return {"status": "error", "message": str(e)}

    def _remove_low_confidence_faces(self, mesh, threshold=0.3):
        """
        Remove faces in low-confidence regions.
        
        Projects each face center onto the 2D confidence map.
        If confidence < threshold, mark face for removal.
        """
        confidence = self.features["confidence"]
        alpha = self.features["alpha"]
        h, w = confidence.shape[:2]

        # Get face centers in 3D
        face_centers = mesh.triangles_center  # (N, 3)
        
        if len(face_centers) == 0:
            return 0

        # Normalize 3D coords to [0, 1] range for projection
        bounds = mesh.bounds  # (2, 3) = [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        extent = bounds[1] - bounds[0]
        extent[extent == 0] = 1.0  # Avoid division by zero

        # Simple orthographic projection: X→u, Y→v (ignore Z for 2D lookup)
        normalized = (face_centers - bounds[0]) / extent
        
        # Map to pixel coords (flip Y for image space)
        u = np.clip((normalized[:, 0] * (w - 1)).astype(int), 0, w - 1)
        v = np.clip(((1.0 - normalized[:, 1]) * (h - 1)).astype(int), 0, h - 1)

        # Lookup confidence for each face
        face_confidence = confidence[v, u]
        face_alpha = alpha[v, u]

        # Only remove faces that are in low-confidence AND low-alpha regions
        # This is conservative to avoid removing valid geometry
        mask_remove = (face_confidence < threshold) & (face_alpha < 64)
        
        faces_to_remove = np.where(mask_remove)[0]
        
        if len(faces_to_remove) > 0 and len(faces_to_remove) < len(mesh.faces) * 0.5:
            # Don't remove more than 50% of faces — safety check
            mask_keep = np.ones(len(mesh.faces), dtype=bool)
            mask_keep[faces_to_remove] = False
            mesh.update_faces(mask_keep)
            mesh.remove_unreferenced_vertices()
            print(f"[MeshRefiner] Removed {len(faces_to_remove)} low-confidence faces")
            return len(faces_to_remove)
        
        return 0

    def _smooth_normals(self, mesh):
        """
        Blend mesh vertex normals with estimated normal map from Stage 3.
        
        Uses a gentle blend (30% estimated, 70% mesh original) to 
        smooth without destroying mesh detail.
        """
        normal_map = self.features["normal"]
        alpha = self.features["alpha"]
        h, w = normal_map.shape[:2]

        if not hasattr(mesh, 'vertex_normals') or mesh.vertex_normals is None:
            return False

        try:
            bounds = mesh.bounds
            extent = bounds[1] - bounds[0]
            extent[extent == 0] = 1.0

            normalized = (mesh.vertices - bounds[0]) / extent

            u = np.clip((normalized[:, 0] * (w - 1)).astype(int), 0, w - 1)
            v = np.clip(((1.0 - normalized[:, 1]) * (h - 1)).astype(int), 0, h - 1)

            # Get estimated normals from normal map (convert from [0,255] to [-1,1])
            est_normals_bgr = normal_map[v, u].astype(np.float32) / 255.0 * 2.0 - 1.0
            # Convert from BGR (OpenCV) to XYZ
            est_normals = np.column_stack([
                est_normals_bgr[:, 2],   # R → X
                est_normals_bgr[:, 1],   # G → Y  
                est_normals_bgr[:, 0],   # B → Z
            ])

            # Only blend where alpha is significant
            vertex_alpha = alpha[v, u].astype(np.float32) / 255.0
            blend_mask = vertex_alpha > 0.5

            if np.sum(blend_mask) > 0:
                blend_weight = 0.3  # Conservative: 30% estimated, 70% original
                blended = mesh.vertex_normals.copy()
                blended[blend_mask] = (
                    mesh.vertex_normals[blend_mask] * (1 - blend_weight) +
                    est_normals[blend_mask] * blend_weight
                )
                # Re-normalize
                norms = np.linalg.norm(blended, axis=1, keepdims=True)
                norms[norms == 0] = 1.0
                blended = blended / norms

                # trimesh stores vertex normals as a property, update in place
                mesh.vertex_normals = blended
                print(f"[MeshRefiner] Smoothed normals for {np.sum(blend_mask)} vertices (30% blend)")
                return True

        except Exception as e:
            print(f"[MeshRefiner] Normal smoothing skipped: {e}")

        return False

    def _correct_depth_conservative(self, mesh):
        """
        Conservative depth correction: nudge vertex Z values toward 
        the monocular depth estimate, but only slightly.
        
        Because monocular depth is relative (not metric), we:
        1. Normalize both mesh Z and estimated depth to [0,1]
        2. Compute difference
        3. Apply a very gentle correction (10% of difference)
        """
        depth_map = self.features["depth"]
        alpha = self.features["alpha"]
        visibility = self.features["visibility"]
        h, w = depth_map.shape[:2]

        try:
            bounds = mesh.bounds
            extent = bounds[1] - bounds[0]
            extent[extent == 0] = 1.0

            normalized = (mesh.vertices - bounds[0]) / extent

            u = np.clip((normalized[:, 0] * (w - 1)).astype(int), 0, w - 1)
            v = np.clip(((1.0 - normalized[:, 1]) * (h - 1)).astype(int), 0, h - 1)

            # Get estimated depth and visibility per vertex
            est_depth_raw = depth_map[v, u].astype(np.float32)
            vert_visibility = visibility[v, u]
            vert_alpha = alpha[v, u].astype(np.float32) / 255.0

            # Normalize estimated depth to [0, 1]
            d_min, d_max = np.min(est_depth_raw[vert_alpha > 0.5]), np.max(est_depth_raw[vert_alpha > 0.5])
            if d_max - d_min <= 0:
                return 0
            est_depth_norm = (est_depth_raw - d_min) / (d_max - d_min)

            # Current mesh Z normalized
            mesh_z_norm = normalized[:, 2]

            # Only correct vertices with high visibility AND high alpha
            correction_mask = (vert_visibility > 0.6) & (vert_alpha > 0.5)

            if np.sum(correction_mask) == 0:
                return 0

            # Compute depth difference
            z_diff = est_depth_norm - mesh_z_norm

            # Apply very gentle correction: 10% of difference
            correction_strength = 0.1
            z_correction = z_diff * correction_strength * correction_mask

            # Scale back to mesh space
            mesh.vertices[:, 2] += z_correction * extent[2]

            corrected_count = int(np.sum(np.abs(z_correction) > 0.001))
            if corrected_count > 0:
                print(f"[MeshRefiner] Adjusted depth for {corrected_count} vertices (10% correction)")

            return corrected_count

        except Exception as e:
            print(f"[MeshRefiner] Depth correction skipped: {e}")
            return 0


def check_quality_gate(quality_json_path: str, threshold: float = 0.85) -> dict:
    """
    Quality gate: Check if conditioning quality score passes threshold.
    
    Returns:
        dict: {"pass": bool, "score": float, "status": str, "details": dict}
    """
    if not os.path.exists(quality_json_path):
        return {"pass": True, "score": None, "status": "no_quality_file", "details": {}}

    with open(quality_json_path, 'r') as f:
        quality = json.load(f)

    score = quality.get("overall_score", 0)
    status = quality.get("status", "UNKNOWN")

    return {
        "pass": score >= threshold,
        "score": score,
        "status": status,
        "details": quality
    }
