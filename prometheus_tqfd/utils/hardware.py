import torch
import psutil

def detect_hardware():
    hw = {
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'gpu_name': None,
        'vram_gb': 0.0,
        'ram_gb': psutil.virtual_memory().total / (1024**3),
        'cpu_cores': psutil.cpu_count(),
        'is_colab': 'google.colab' in str(get_ipython()) if 'get_ipython' in globals() else False
    }

    if hw['device'] == 'cuda':
        hw['gpu_name'] = torch.cuda.get_device_name(0)
        hw['vram_gb'] = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    return hw
