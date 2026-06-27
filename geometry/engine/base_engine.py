import abc

class Image3DEngine(abc.ABC):
    """
    Base Interface for all Image-to-3D Engines (StableFast3D, Hunyuan3D, etc.)
    Ensures that the entire pipeline can easily swap models without changing UI/API.
    """
    
    @abc.abstractmethod
    def generate3D(self, input_path: str, output_dir: str) -> dict:
        """
        Generate 3D mesh from input.
        
        Args:
            input_path (str): Path to the input image or geometry feature (e.g. .npz, .png)
            output_dir (str): Path to the directory where the output meshes will be saved.
            
        Returns:
            dict: Status and path information of the generated files.
                  Example: {"status": "success", "obj_path": "...", "glb_path": "..."}
        """
        pass
