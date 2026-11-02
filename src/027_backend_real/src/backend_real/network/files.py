import os

# FILE_MARKERS_CSV = Path.cwd().glob("markers.csv").__next__()
# FILE_CAM_CONFIG_YAML = Path.cwd().glob("calibration.yaml").__next__()
FILE_MARKERS_CSV = os.environ["FILE_MARKERS_CSV"]
FILE_CAM_CONFIG_YAML = os.environ["FILE_CAM_CONFIG_YAML"]
