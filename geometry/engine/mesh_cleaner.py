import trimesh

class MeshCleaner:
    def __init__(self, mesh_path):
        self.mesh_path = mesh_path
        self.mesh = trimesh.load(mesh_path)

    def clean(self):
        print("[MeshCleaner] Cleaning mesh...")
        self.mesh.remove_duplicate_faces()
        self.mesh.remove_unreferenced_vertices()
        self.mesh.fill_holes()
        self.mesh.export(self.mesh_path)
        print("[MeshCleaner] Mesh cleaned.")
