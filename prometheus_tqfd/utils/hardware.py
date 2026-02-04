import torch
import psutil
from prometheus_tqfd.config import HardwareConfig

def detect_hardware() -> HardwareConfig:
    """
    Erkennt Hardware und setzt optimale Parameter.
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    gpu_name = None
    vram_gb = None

    if device == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)

    ram_gb = psutil.virtual_memory().total / (1024**3)
    cpu_cores = psutil.cpu_count()

    # Colab Check
    is_colab = False
    try:
        import google.colab
        is_colab = True
    except ImportError:
        pass

    return HardwareConfig(
        device=device,
        gpu_name=gpu_name,
        vram_gb=vram_gb,
        ram_gb=ram_gb,
        cpu_cores=cpu_cores,
        is_colab=is_colab
    )
