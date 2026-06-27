import trimesh

class MeshValidator:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.mesh = trimesh.load(mesh_path)

    def validate(self):
        print("[MeshValidator] Validating mesh...")
        results = {
            "vertices": len(self.mesh.vertices),
            "faces": len(self.mesh.faces),
            "is_watertight": self.mesh.is_watertight,
            "bounds": self.mesh.bounds.tolist() if self.mesh.bounds is not None else []
        }
        print("[MeshValidator] Validation complete.")
        return results
