import yaml
import sys
from pathlib import Path
import os

# === 设置路径 ===
FILE = Path(__file__).resolve()
PROJECT_ROOT = FILE.parents[1]
sys.path.append(str(PROJECT_ROOT))


def load_settings(path='config/global_setting.yaml'):
    path = os.path.abspath(os.path.join(PROJECT_ROOT, path))
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

config = load_settings()

def get_model_weight_path():
    return config['model']['weights']

def get_input_size():
    return config['model']['input_size']

def get_threshold():
    return config['model']['conf_threshold']

def get_draw_radius():
    return config['draw']['radius']

if __name__ == "__main__":
    # Load settings
    config = load_settings()

    # Access parameters
    weights_path = config['model']['weights']
    input_size = config['model']['input_size']
    threshold = config['model']['conf_threshold']

    test_folder = config['images']['test_folder']
    save_folder = config['images']['save_folder']

    board_points = config['board_points']

    inner_bull = config['circle_radius']['inner_bull']
    outer_bull = config['circle_radius']['outer_bull']
    triple_ring = config['circle_radius']['triple_ring']
    double_ring = config['circle_radius']['double_ring']

    # Example usage
    print(f"Model weights path: {weights_path}")
    print(f"Triple ring range: {triple_ring}")
