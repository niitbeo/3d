import json
import os

class ReportExporter:
    def __init__(self, val_results, inference_time, output_dir):
        self.val_results = val_results
        self.inference_time = inference_time
        self.output_dir = output_dir

    def export(self):
        print("[ReportExporter] Generating mesh report...")
        report_path = os.path.join(self.output_dir, "mesh_report.json")
        
        report = {
            "engine": "Stable Fast 3D",
            "device": "Apple MPS",
            "inference_time": round(self.inference_time, 2),
            "vertices": self.val_results.get("vertices", 0),
            "faces": self.val_results.get("faces", 0),
            "is_watertight": self.val_results.get("is_watertight", False),
            "bounds": self.val_results.get("bounds", []),
            "status": "PASS" if self.val_results.get("vertices", 0) > 0 else "FAIL"
        }
        
        with open(report_path, "w") as f:
            json.dump(report, f, indent=4)
            
        print("[ReportExporter] Report generated.")
        return report_path
