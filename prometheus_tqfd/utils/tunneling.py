import os
import subprocess
import time
import re
from typing import Optional, Tuple

class TunnelManager:
    """
    Verwaltet öffentlichen Zugang zum Dashboard.
    Priorität: ngrok → cloudflared → localtunnel
    """

    @staticmethod
    def start(port: int = 8501) -> Tuple[Optional[str], str]:
        # 1. Try ngrok
        url = TunnelManager._try_ngrok(port)
        if url:
            return url, 'ngrok'

        # 2. Try cloudflared
        url = TunnelManager._try_cloudflared(port)
        if url:
            return url, 'cloudflared'

        # 3. Try localtunnel
        url = TunnelManager._try_localtunnel(port)
        if url:
            return url, 'localtunnel'

        return None, 'none'

    @staticmethod
    def _try_ngrok(port: int) -> Optional[str]:
        try:
            from pyngrok import ngrok

            # Token detection
            token = os.environ.get('NGROK_TOKEN')
            if not token:
                try:
                    from google.colab import userdata
                    token = userdata.get('NGROK_TOKEN')
                except:
                    pass

            if token:
                ngrok.set_auth_token(token)

            tunnel = ngrok.connect(port, "http")
            print(f"✅ ngrok Tunnel: {tunnel.public_url}")
            return tunnel.public_url
        except Exception as e:
            # print(f"⚠️ ngrok failed: {e}")
            return None

    @staticmethod
    def _try_cloudflared(port: int) -> Optional[str]:
        try:
            # Check if cloudflared exists
            if not os.path.exists('cloudflared'):
                subprocess.run([
                    'wget', '-q',
                    'https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64',
                    '-O', 'cloudflared'
                ], check=True)
                subprocess.run(['chmod', '+x', 'cloudflared'], check=True)

            proc = subprocess.Popen(
                ['./cloudflared', 'tunnel', '--url', f'http://localhost:{port}'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            # Non-blocking wait for URL
            timeout = 15
            start_time = time.time()
            while time.time() - start_time < timeout:
                line = proc.stderr.readline()
                if 'trycloudflare.com' in line:
                    match = re.search(r'https://[^\s]+\.trycloudflare\.com', line)
                    if match:
                        url = match.group(0)
                        print(f"✅ cloudflared Tunnel: {url}")
                        return url
                time.sleep(0.1)
            return None
        except Exception as e:
            # print(f"⚠️ cloudflared failed: {e}")
            return None

    @staticmethod
    def _try_localtunnel(port: int) -> Optional[str]:
        try:
            # Requires npx / node
            proc = subprocess.Popen(
                ['npx', 'localtunnel', '--port', str(port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            timeout = 10
            start_time = time.time()
            while time.time() - start_time < timeout:
                line = proc.stdout.readline()
                if 'your url is:' in line.lower():
                    url = line.split()[-1]
                    print(f"✅ localtunnel: {url}")
                    return url
                time.sleep(0.1)
            return None
        except Exception as e:
            # print(f"⚠️ localtunnel failed: {e}")
            return None

def setup_tunnel(port, ngrok_token=None):
    if ngrok_token:
        os.environ['NGROK_TOKEN'] = ngrok_token
    url, method = TunnelManager.start(port)
    return url
