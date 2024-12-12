import argparse
import os

def parse_arguments(
        default_config_path = None, 
        default_results_path = None, 
        default_calibrate_images_folder = None
        ):
    parser = argparse.ArgumentParser(prog="Camera calibrator", usage="%(prog)s [OPTIONS]")
    parser.add_argument(
        "--config-path", 
        type=argparse.FileType("r"), 
        help="Path to config file", 
        default=default_config_path
        )
    
    parser.add_argument(
        "--results", 
        type=str, 
        help="Path to output folder",
        default="data/results/"
        )
    
    parser.add_argument(
        "--images", 
        type=str, 
        help="Path to calibrate images",
        default=default_calibrate_images_folder
        )
    
    parser.add_argument("--config", help="Path to config file")
    args = parser.parse_args()
    return args