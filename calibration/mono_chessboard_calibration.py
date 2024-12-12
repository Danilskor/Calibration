import cv2
import numpy as np
from calibration.camera_calibration_base import CameraCalibrationBase

class MonoChessboardCalibration(CameraCalibrationBase):
    def __init__(self, config):
        """
        Класс для выполнения моно калибровки с использованием шахматной доски.

        :param config: конфигурация с параметрами шахматной доски и камеры.
        """
        super().__init__(calibration_type="monocular", config=config)
        self.board_size = (config['x_size'], config['y_size'])
        self.square_size = config['square_length']
        self.object_points = self._prepare_object_points()

    def _prepare_object_points(self):
        """
        Создаёт 3D точки шахматной доски в реальном пространстве.

        :return: массив 3D точек.
        """
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        return objp * self.square_size

    def detect_features(self, images):
        """
        Обнаруживает углы шахматной доски на изображениях.

        :param images: список изображений.
        :return: списки точек изображения и соответствующих 3D точек.
        """
        obj_points = []  # 3D точки реального мира.
        img_points = []  # 2D точки на изображениях.

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, self.board_size, None)

            if found:
                # Улучшение точности углов
                corners_subpix = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1),
                    criteria=(cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 30, 0.001)
                )
                img_points.append(corners_subpix)
                obj_points.append(self.object_points)
        return obj_points, img_points

    def calibrate(self, obj_points, img_points, image_size):
        """
        Выполняет калибровку камеры.

        :param obj_points: список 3D точек.
        :param img_points: список 2D точек.
        :param image_size: размер изображений (ширина, высота).
        :return: среднеквадратическая ошибка калибровки.
        """
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            obj_points, img_points, image_size, None, None
        )
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs
        self.rvecs = rvecs
        self.tvecs = tvecs
        return ret

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
