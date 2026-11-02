from pathlib import Path

FILE_MARKERS_CSV = Path.cwd().glob("markers.csv").__next__()
FILE_CAM_CONFIG_YAML = Path.cwd().glob("calibration.yaml").__next__()
