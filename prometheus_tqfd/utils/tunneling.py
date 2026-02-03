import os
import subprocess
import time

class TunnelManager:
    @staticmethod
    def start_ngrok(port, token=None):
        try:
            from pyngrok import ngrok
            if token:
                ngrok.set_auth_token(token)
            tunnel = ngrok.connect(port, "http")
            return tunnel.public_url
        except Exception as e:
            print(f"ngrok failed: {e}")
            return None

    @staticmethod
    def start_cloudflared(port):
        try:
            # Download if not present
            if not os.path.exists("./cloudflared"):
                subprocess.run(["wget", "-q", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"])
                subprocess.run(["chmod", "+x", "cloudflared"])

            process = subprocess.Popen(["./cloudflared", "tunnel", "--url", f"http://localhost:{port}"],
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

            # Wait for URL to appear in output
            for _ in range(30):
                line = process.stdout.readline()
                if ".trycloudflare.com" in line:
                    url = line.strip().split()[-1]
                    # Sometimes URL is inside a sentence
                    for word in line.split():
                        if ".trycloudflare.com" in word:
                            return word
                time.sleep(1)
            return None
        except Exception as e:
            print(f"cloudflared failed: {e}")
            return None

    @staticmethod
    def start_localtunnel(port):
        try:
            process = subprocess.Popen(["npx", "localtunnel", "--port", str(port)],
                                     stdout=subprocess.PIPE, text=True)
            line = process.stdout.readline()
            if "your url is:" in line.lower():
                return line.split()[-1]
            return None
        except Exception as e:
            print(f"localtunnel failed: {e}")
            return None

def setup_tunnel(port, ngrok_token=None):
    # 1. ngrok
    if ngrok_token:
        url = TunnelManager.start_ngrok(port, ngrok_token)
        if url: return url

    # 2. cloudflared
    url = TunnelManager.start_cloudflared(port)
    if url: return url

    # 3. localtunnel
    url = TunnelManager.start_localtunnel(port)
    return url
