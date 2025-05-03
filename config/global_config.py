import yaml

def load_settings(path='setting.yaml'):
    with open(path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

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
