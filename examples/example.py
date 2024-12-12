import pathlib
from scripts import argpaser
from calibration import utils

DEFAULT_CONFIG_PATH = "calibration/configs/default.yml"
DEFAULT_CALIBRATE_IMAGES_FOLDER = "data/images"

args = argpaser.parse_arguments(
    default_config_path = DEFAULT_CONFIG_PATH, 
    default_calibrate_images_folder = DEFAULT_CALIBRATE_IMAGES_FOLDER,
    )

images_folder_path = pathlib.Path(args.images)
images_paths_list = utils.get_images_paths(images_folder_path)
images = utils.get_images(images_paths_list)