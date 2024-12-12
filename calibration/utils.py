import glob
import cv2
import pathlib
import logging

def get_images_paths(folder_path: pathlib.Path):
    images_paths_list = list(folder_path.resolve().rglob("*.*"))
    print(folder_path.resolve())
    if not images_paths_list:
        raise FileNotFoundError(f"No files found in directory: {folder_path}")
    return images_paths_list

def get_images(images_path_list: list[pathlib.Path]):
    images = [] 
    for image_path in images_path_list:
        image = cv2.imread(image_path)
        if image is None:
            logging.warning(f"File {image_path} didn't load")
            continue
        images.append(image)
    if not images:
        raise FileNotFoundError("Not a single image has loaded")
    return images
