import cv2
import numpy as np
from calibration.camera_calibration_base import CameraCalibrationBase

class MonoCharucoCalibration(CameraCalibrationBase):
    def __init__(self, config):
        """
        Класс для выполнения моно калибровки с использованием ChArUco доски.
        
        :param config: конфигурация с параметрами ChArUco доски и камеры.
        """
        super().__init__(calibration_type="monocular", config=config)
        self.charuco_dict = cv2.aruco.Dictionary_get(config['aruco_dict'])
        self.charuco_board = cv2.aruco.CharucoBoard_create(
            config['squares_x'], config['squares_y'], 
            config['square_length'], config['marker_length'], self.charuco_dict
        )

    def detect_features(self, images):
        """
        Обнаруживает углы ChArUco доски на изображениях.
        :param images: список изображений.
        :return: список пар (corners, ids) для каждого изображения.
        """
        all_corners = []
        all_ids = []
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.charuco_dict)
            if len(corners) > 0:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    corners, ids, gray, self.charuco_board
                )
                if charuco_corners is not None and charuco_ids is not None:
                    all_corners.append(charuco_corners)
                    all_ids.append(charuco_ids)
        return all_corners, all_ids

    def calibrate(self, corners, ids, image_size):
        """
        Выполняет калибровку на основе углов и ID ChArUco.
        :param corners: список углов ChArUco.
        :param ids: список ID для углов.
        :param image_size: размер изображений (ширина, высота).
        """
        flags = cv2.CALIB_USE_INTRINSIC_GUESS if 'camera_matrix' in self.config else 0
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.aruco.calibrateCameraCharuco(
            corners, ids, self.charuco_board, image_size, 
            self.camera_matrix, self.dist_coeffs, flags=flags
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        return ret

    def draw_board(self, output_path):
        """
        Генерирует изображение ChArUco доски.
        :param output_path: путь для сохранения изображения.
        """
        board_image = self.charuco_board.draw((600, 600))
        cv2.imwrite(output_path, board_image)

    def undistort_image(self, image):
        """
        Применяет коррекцию искажений к изображению.
        :param image: исходное изображение.
        :return: скорректированное изображение.
        """
        h, w = image.shape[:2]
        new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            self.camera_matrix, self.dist_coeffs, (w, h), 1, (w, h)
        )
        return cv2.undistort(image, self.camera_matrix, self.dist_coeffs, None, new_camera_matrix)
