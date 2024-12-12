import numpy as np
import cv2
import os 
import yaml

class CameraCalibrationBase:
    def __init__(self, calibration_type, config=None):
        """
        Базовый класс для калибровки камеры.
        """
        self.calibration_type = calibration_type
        self.config = config or {}
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None

    def load_images(self, image_paths):
        """
        Загружает изображения из списка путей.
        
        :param image_paths: список путей к изображениям.
        :return: список изображений.
        """
        images = []
        for path in image_paths:
            if not os.path.exists(path):
                print(f"Warning: Image path does not exist: {path}")
                continue
            img = cv2.imread(path)
            if img is None:
                print(f"Warning: Failed to load image at path: {path}")
            else:
                images.append(img)
        return images

    def detect_features(self, images):
        raise NotImplementedError("Метод должен быть реализован в подклассе.")

    def calibrate(self, features):
        raise NotImplementedError("Метод должен быть реализован в подклассе.")

    def save_calibration(self, file_path):
        data = {
            "camera_matrix": self.camera_matrix.tolist() if self.camera_matrix is not None else None,
            "dist_coeffs": self.dist_coeffs.tolist() if self.dist_coeffs is not None else None,
            "rvecs": [r.tolist() for r in self.rvecs] if self.rvecs else None,
            "tvecs": [t.tolist() for t in self.tvecs] if self.tvecs else None,
        }
        with open(file_path, "w") as f:
            yaml.dump(data, f, sort_keys=False)

    def load_calibration(self, file_path):
        with open(file_path, "r") as f:
            data = yaml.safe_load(f)
        self.camera_matrix = np.array(data["camera_matrix"]) if data["camera_matrix"] else None
        self.dist_coeffs = np.array(data["dist_coeffs"]) if data["dist_coeffs"] else None
        self.rvecs = [np.array(r) for r in data["rvecs"]] if data["rvecs"] else None
        self.tvecs = [np.array(t) for t in data["tvecs"]] if data["tvecs"] else None