import os
import sys
import subprocess
import time
import multiprocessing as mp
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

def install_dependencies():
    print("📦 Installing dependencies...")
    libs = [
        "python-chess", "numpy", "torch", "einops", "psutil",
        "lz4", "safetensors", "plotly", "streamlit", "pyngrok"
    ]
    # Check if torch_geometric can be installed easily
    try:
        subprocess.run(["pip", "install", "--quiet"] + libs, check=True)
    except Exception as e:
        print(f"Warning: Error during installation: {e}")

def setup_directories(config):
    # Handle /content for Colab
    if os.path.exists("/content"):
        config.base_dir = Path("/content/prometheus_runs")

    run_dir = config.base_dir / config.run_id
    for sub in ['checkpoints', 'metrics', 'games', 'logs']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

    # Create a link to latest
    latest_link = config.base_dir / 'latest'
    if latest_link.exists():
        if latest_link.is_symlink(): latest_link.unlink()
        else: shutil.rmtree(latest_link)

    # In Colab, we can't always symlink. Let's just use it as a name.
    # We'll handle 'latest' logic in CP manager.
    pass

def start_dashboard(port):
    print(f"🚀 Starting Streamlit Dashboard on port {port}...")
    # Start streamlit as a subprocess
    process = subprocess.Popen([
        sys.executable, "-m", "streamlit", "run",
        "prometheus_tqfd/dashboard/app.py",
        f"--server.port={port}",
        "--server.headless=true"
    ])
    return process

def main():
    print("🔥 PROMETHEUS-TQFD Dual Chess AI Training System")

    # mp setup
    mp.set_start_method('spawn', force=True)

    # 1. Hardware Detection
    from prometheus_tqfd.utils.hardware import detect_hardware
    hw = detect_hardware()
    print(f"📊 Hardware: {hw['device'].upper()} - {hw['gpu_name']}")

    # 2. Config
    from prometheus_tqfd.config import PrometheusConfig, adjust_config_for_hardware
    config = PrometheusConfig()

    # Google Drive Mount
    if os.path.exists("/content") and config.use_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
        except:
            print("⚠️ Could not mount Google Drive.")
    config = adjust_config_for_hardware(config)

    # 3. Setup Directories
    setup_directories(config)

    # 4. Start Dashboard & Tunnel
    dashboard_proc = start_dashboard(config.dashboard_port)
    from prometheus_tqfd.utils.tunneling import setup_tunnel
    # Try to get ngrok token from environment or secrets
    ngrok_token = os.environ.get('NGROK_TOKEN')
    try:
        from google.colab import userdata
        ngrok_token = ngrok_token or userdata.get('NGROK_TOKEN')
    except:
        pass

    public_url = setup_tunnel(config.dashboard_port, ngrok_token=ngrok_token)
    print(f"✅ Dashboard URL: {public_url}")

    # 5. Start Supervisor
    from prometheus_tqfd.orchestration.supervisor import Supervisor
    supervisor = Supervisor(config)

    print("\n🚀 Starting Training...")
    try:
        supervisor.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested.")
    finally:
        if dashboard_proc: dashboard_proc.terminate()
        print("✅ Prometheus finished.")

if __name__ == "__main__":
    main()
