"""
Download PnLCalib weights reliably using urllib with chunked streaming.
"""
import os
import sys
import time
import urllib.request

WEIGHTS = {
    "SV_kp": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
    "SV_lines": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
}

def download_file(url: str, output_path: str, retries: int = 5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Downloading {os.path.basename(output_path)} (Attempt {attempt}/{retries})...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=30) as response, open(output_path, 'wb') as out_file:
                total_size = response.getheader('Content-Length')
                if total_size is not None:
                    total_size = int(total_size)
                downloaded = 0
                block_size = 1024 * 1024  # 1 MB blocks
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    if total_size:
                        percent = downloaded / total_size * 100
                        print(f" -> Downloaded {downloaded / (1024*1024):.2f} MB / {total_size / (1024*1024):.2f} MB ({percent:.1f}%)", end='\r')
            print(f"\n[SUCCESS] Downloaded {os.path.basename(output_path)} ({os.path.getsize(output_path) / (1024*1024):.2f} MB)")
            return True
        except Exception as e:
            print(f"\n[WARNING] Attempt {attempt} failed: {e}")
            time.sleep(3)
    return False

def main():
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    for name, url in WEIGHTS.items():
        out_file = os.path.join(weights_dir, name)
        if os.path.exists(out_file) and os.path.getsize(out_file) > 10 * 1024 * 1024:
            print(f"[INFO] {name} already downloaded ({os.path.getsize(out_file)/(1024*1024):.2f} MB). Skipping.")
            continue
        download_file(url, out_file)

if __name__ == "__main__":
    main()
