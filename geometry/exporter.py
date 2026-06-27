import os
import trimesh

def export_mesh(mesh, output_dir, base_name):
    print(f"[Exporter] Exporting mesh to {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Export GLB (Best for WebAR / Three.js)
    glb_path = os.path.join(output_dir, f"{base_name}_coarse_mesh.glb")
    mesh.export(glb_path, include_normals=True)
    print(f"[Exporter] Saved {glb_path}")
    
    # Export OBJ
    obj_path = os.path.join(output_dir, f"{base_name}.obj")
    mesh.export(obj_path, include_normals=True)
    print(f"[Exporter] Saved {obj_path}")
    
    # Export STL (Best for 3D Printing / Resin)
    stl_path = os.path.join(output_dir, f"{base_name}.stl")
    mesh.export(stl_path)
    print(f"[Exporter] Saved {stl_path}")
    
    # Texture is saved automatically alongside OBJ/GLB by trimesh if it has visuals
    # Preview PNG
    preview_path = os.path.join(output_dir, f"{base_name}_preview.png")
    try:
        scene = mesh.scene()
        png_data = scene.save_image(resolution=(512, 512), visible=True)
        if png_data:
            with open(preview_path, 'wb') as f:
                f.write(png_data)
            print(f"[Exporter] Saved preview {preview_path}")
    except Exception as e:
        print(f"[Exporter] Skipping preview PNG generation (OpenGL/pyrender issue): {e}")

    print("[Exporter] Export COMPLETE.")
