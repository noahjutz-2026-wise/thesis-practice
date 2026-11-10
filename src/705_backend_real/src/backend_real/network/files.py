import os


def get_file_markers_csv():
    return os.environ.get("FILE_MARKERS_CSV", default="/data/markers.csv")


def get_file_cam_config_yaml():
    return os.environ.get("FILE_CAM_CONFIG_YAML", default="/data/calibration.yaml")
