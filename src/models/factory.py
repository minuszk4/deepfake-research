import timm
from .mesonet import Meso4

def create_model(model_key, config, num_classes=1):
    """
    Khởi tạo mô hình dựa trên cấu hình YAML.
    """
    if config['type'] == 'custom' and model_key == 'meso4':
        return Meso4(num_classes=num_classes)
    elif config['type'] == 'timm':
        return timm.create_model(config['name'], pretrained=True, num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model type or key: {config['type']} / {model_key}")
