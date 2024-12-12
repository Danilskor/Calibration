import cv2
import numpy as np
from calibration.camera_calibration_base import CameraCalibrationBase

class StereoChessboardCalibration(CameraCalibrationBase):
    def __init__(self, config):
        """
        Класс для выполнения стереокалибровки с использованием шахматной доски.

        :param config: конфигурация с параметрами шахматной доски и камеры.
        """
        super().__init__(calibration_type="stereo", config=config)
        self.board_size = (config['x_size'], config['y_size'])
        self.square_size = config['square_length']
        self.object_points = self._prepare_object_points()
        
        self.stereo_params = None

    def _prepare_object_points(self):
        """
        Создаёт 3D точки шахматной доски в реальном пространстве.

        :return: массив 3D точек.
        """
        objp = np.zeros((self.board_size[0] * self.board_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.board_size[0], 0:self.board_size[1]].T.reshape(-1, 2)
        return objp * self.square_size

    def detect_features(self, left_images, right_images):
        """
        Обнаруживает углы шахматной доски на паре изображений.

        :param left_images: список изображений с левой камеры.
        :param right_images: список изображений с правой камеры.
        :return: списки точек изображения и соответствующих 3D точек для обеих камер.
        """
        obj_points = []  # 3D точки реального мира.
        img_points_left = []  # 2D точки на изображениях левой камеры.
        img_points_right = []  # 2D точки на изображениях правой камеры.

        for left_img, right_img in zip(left_images, right_images):
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)

            found_left, corners_left = cv2.findChessboardCorners(gray_left, self.board_size, None)
            found_right, corners_right = cv2.findChessboardCorners(gray_right, self.board_size, None)

            if found_left and found_right:
                criteria=cv2.TermCriteria_EPS + cv2.TermCriteria_COUNT, 30, 0.001
                corners_left = cv2.cornerSubPix(
                    gray_left, corners_left, (11, 11), (-1, -1),
                    criteria=criteria
                )
                corners_right = cv2.cornerSubPix(
                    gray_right, corners_right, (11, 11), (-1, -1),
                    criteria=criteria
                )

                obj_points.append(self.object_points)
                img_points_left.append(corners_left)
                img_points_right.append(corners_right)

        return obj_points, img_points_left, img_points_right

    def calibrate(self, obj_points, img_points_left, img_points_right, image_size):
        """
        Выполняет стереокалибровку камер.

        :param obj_points: список 3D точек.
        :param img_points_left: список 2D точек на изображениях левой камеры.
        :param img_points_right: список 2D точек на изображениях правой камеры.
        :param image_size: размер изображений (ширина, высота).
        :return: среднеквадратическая ошибка калибровки.
        """
        # Калибровка для левой камеры
        _, camera_matrix_left, dist_coeffs_left, _, _ = cv2.calibrateCamera(
            obj_points, img_points_left, image_size, None, None
        )

        # Калибровка для правой камеры
        _, camera_matrix_right, dist_coeffs_right, _, _ = cv2.calibrateCamera(
            obj_points, img_points_right, image_size, None, None
        )

        # Стереокалибровка
        rms_error, _, _, _, _, R, T, E, F = cv2.stereoCalibrate(
            obj_points, img_points_left, img_points_right,
            camera_matrix_left, dist_coeffs_left,
            camera_matrix_right, dist_coeffs_right,
            image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-5),
            flags=cv2.CALIB_FIX_INTRINSIC
        )

        self.stereo_params = {
            'camera_matrix_left': camera_matrix_left,
            'dist_coeffs_left': dist_coeffs_left,
            'camera_matrix_right': camera_matrix_right,
            'dist_coeffs_right': dist_coeffs_right,
            'R': R,
            'T': T,
            'E': E,
            'F': F
        }
        return rms_error

    def rectify(self, image_size):
        """
        Выполняет выравнивание изображений на основе результатов стереокалибровки.

        :param image_size: размер изображений (ширина, высота).
        :return: параметры выравнивания (map1, map2 для обеих камер).
        """
        if not self.stereo_params:
            raise ValueError("Стереокалибровка не выполнена. Пожалуйста, выполните calibrate().")

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            self.stereo_params['camera_matrix_left'], self.stereo_params['dist_coeffs_left'],
            self.stereo_params['camera_matrix_right'], self.stereo_params['dist_coeffs_right'],
            image_size, self.stereo_params['R'], self.stereo_params['T'], alpha=0
        )

        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.stereo_params['camera_matrix_left'], self.stereo_params['dist_coeffs_left'],
            R1, P1, image_size, cv2.CV_16SC2
        )
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.stereo_params['camera_matrix_right'], self.stereo_params['dist_coeffs_right'],
            R2, P2, image_size, cv2.CV_16SC2
        )

        return (map1_left, map2_left), (map1_right, map2_right)

    def undistort_and_rectify(self, left_image, right_image, rectification_maps):
        """
        Применяет коррекцию и выравнивание к паре изображений.

        :param left_image: изображение с левой камеры.
        :param right_image: изображение с правой камеры.
        :param rectification_maps: параметры выравнивания (карты).
        :return: выровненные изображения.
        """
        (map1_left, map2_left), (map1_right, map2_right) = rectification_maps

        rectified_left = cv2.remap(left_image, map1_left, map2_left, cv2.INTER_LINEAR)
        rectified_right = cv2.remap(right_image, map1_right, map2_right, cv2.INTER_LINEAR)

        return rectified_left, rectified_right