import os
import sys
import subprocess
import time
import multiprocessing as mp
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

def setup_directories(config):
    # Handle /content for Colab
    if os.path.exists("/content"):
        config.base_dir = Path("/content/prometheus_runs")
    else:
        config.base_dir = Path("./prometheus_runs")

    run_dir = config.base_dir / config.run_id
    for sub in ['checkpoints', 'metrics', 'games', 'logs']:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)

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
    print("=" * 60)
    print("🔥 PROMETHEUS-TQFD v2.0")
    print("   Dual-AI Tabula Rasa Chess Training System")
    print("=" * 60)

    # 1. mp setup
    mp.set_start_method('spawn', force=True)

    # 2. Hardware Detection
    from prometheus_tqfd.utils.hardware import detect_hardware
    hw = detect_hardware()
    print(f"\n📊 Hardware erkannt:")
    print(f"   Device: {hw.device}")
    if hw.gpu_name:
        print(f"   GPU: {hw.gpu_name} ({hw.vram_gb:.1f} GB VRAM)")
    print(f"   RAM: {hw.ram_gb:.1f} GB")

    # 3. Config
    from prometheus_tqfd.config import PrometheusConfig, adjust_config_for_hardware
    config = PrometheusConfig()
    config = adjust_config_for_hardware(config, hw)

    # 4. Setup Directories
    setup_directories(config)
    run_dir = config.base_dir / config.run_id

    # 5. Smoke Tests
    from prometheus_tqfd.tests import run_smoke_tests
    if not run_smoke_tests():
        print("⛔ Smoke Tests failed. Training aborted.")
        return

    # 6. Google Drive Mount (if in Colab)
    if hw.is_colab and config.use_drive:
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            print("✅ Google Drive gemountet")
        except:
            print("⚠️ Google Drive nicht verfügbar")
            config.use_drive = False

    # 7. Start Dashboard
    dashboard_proc = start_dashboard(config.dashboard_port)

    # 8. Setup Tunnel
    from prometheus_tqfd.utils.tunneling import setup_tunnel
    ngrok_token = os.environ.get('NGROK_TOKEN')
    try:
        from google.colab import userdata
        ngrok_token = ngrok_token or userdata.get('NGROK_TOKEN')
    except:
        pass

    url = setup_tunnel(config.dashboard_port, ngrok_token=ngrok_token)
    if url:
        print(f"🌐 Dashboard URL: {url}")

    # 9. Start Supervisor
    from prometheus_tqfd.orchestration.supervisor import Supervisor
    supervisor = Supervisor(config)

    print("\n🚀 Starting Training Loop...")
    try:
        supervisor.run()
    except KeyboardInterrupt:
        print("\n🛑 Shutdown requested.")
    finally:
        if dashboard_proc: dashboard_proc.terminate()
        print("✅ Prometheus finished.")

if __name__ == "__main__":
    main()
