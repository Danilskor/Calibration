import unittest

from calibration.mono_charuco_calibration import MonoCharucoCalibration

class TestMonoCharucoCalibration(unittest.TestCase):
    def setUp(self):
        self.calibrator = MonoCharucoCalibration()
        return super().setUp()
    



if __name__ == "__main__":
    unittest.main()