import trimesh
import numpy as np

def validate_mesh(mesh):
    print("[MeshValidator] Validating Mesh...")
    is_watertight = mesh.is_watertight
    is_watertight_str = "PASS" if is_watertight else "FAIL (Will auto-repair holes)"
    
    is_winding_consistent = mesh.is_winding_consistent
    winding_str = "PASS" if is_winding_consistent else "FAIL (Normals issue)"
    
    print(f"[MeshValidator] Watertight: {is_watertight_str}")
    print(f"[MeshValidator] Winding Consistent: {winding_str}")
    print(f"[MeshValidator] Bounding Box: {mesh.bounds}")
    return is_watertight, is_winding_consistent

def clean_mesh(mesh):
    print(f"[MeshCleaner] Original Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    
    # Validation before cleaning
    validate_mesh(mesh)
    
    # 1. Remove duplicate and unreferenced vertices
    mesh.remove_duplicate_faces()
    mesh.remove_unreferenced_vertices()
    print("[MeshCleaner] Removed duplicate/unreferenced vertices.")
    
    # 2. Repair holes (if not watertight)
    if not mesh.is_watertight:
        try:
            mesh.fill_holes()
            print("[MeshCleaner] Holes repaired.")
        except Exception as e:
            print(f"[MeshCleaner] Failed to fill holes: {e}")
            
    # 3. Recalculate normals
    mesh.fix_normals()
    print("[MeshCleaner] Normals fixed.")
    
    # 4. Remove tiny islands
    # We keep the largest connected component
    components = mesh.split(only_watertight=False)
    if len(components) > 1:
        # Sort by number of faces
        components.sort(key=lambda c: len(c.faces), reverse=True)
        mesh = components[0] # Keep the largest
        print(f"[MeshCleaner] Removed {len(components) - 1} tiny islands.")
        
    # 5. Smooth slightly (Taubin smoothing preserves volume)
    try:
        from trimesh.smoothing import filter_taubin
        filter_taubin(mesh, iterations=3)
        print("[MeshCleaner] Applied light smoothing.")
    except Exception as e:
        print(f"[MeshCleaner] Smoothing failed: {e}")

    # Validation after cleaning
    validate_mesh(mesh)
    
    print(f"[MeshCleaner] Cleaned Mesh: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    return mesh
