"""
Verify PyTorch weight zip headers and re-download clean weights if corrupted.
"""
import os
import sys
import time
import zipfile
import torch
import urllib.request

WEIGHTS = {
    "SV_kp": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_kp",
    "SV_lines": "https://github.com/mguti97/PnLCalib/releases/download/v1.0.0/SV_lines",
}

def verify_weight_file(path: str) -> bool:
    if not os.path.exists(path):
        print(f"[MISSING] {path} does not exist.")
        return False
    try:
        # Check if valid PyTorch zip archive
        with zipfile.ZipFile(path, 'r') as z:
            bad_file = z.testzip()
            if bad_file is not None:
                print(f"[CORRUPTED] {path} zip test failed on {bad_file}")
                return False
        print(f"[VALID] {os.path.basename(path)} zip header verified ({os.path.getsize(path)/(1024*1024):.2f} MB).")
        return True
    except Exception as e:
        print(f"[CORRUPTED] {path} invalid zip header: {e}")
        return False

def download_clean_file(url: str, output_path: str, retries: int = 5):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    for attempt in range(1, retries + 1):
        try:
            print(f"[INFO] Downloading {os.path.basename(output_path)} (Attempt {attempt}/{retries})...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=60) as response, open(output_path, 'wb') as out_file:
                total_size = response.getheader('Content-Length')
                total_size = int(total_size) if total_size else None
                downloaded = 0
                block_size = 2 * 1024 * 1024  # 2 MB chunks
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    out_file.write(buffer)
                    downloaded += len(buffer)
                    if total_size:
                        print(f" -> {downloaded/(1024*1024):.2f} MB / {total_size/(1024*1024):.2f} MB ({downloaded/total_size*100:.1f}%)", end='\r')
            print(f"\n[SUCCESS] Download finished: {os.path.basename(output_path)}")
            if verify_weight_file(output_path):
                return True
        except Exception as e:
            print(f"\n[WARNING] Attempt {attempt} failed: {e}")
            time.sleep(2)
    return False

def main():
    weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
    for name, url in WEIGHTS.items():
        out_file = os.path.join(weights_dir, name)
        if not verify_weight_file(out_file):
            print(f"[INFO] Re-downloading {name}...")
            download_clean_file(url, out_file)

if __name__ == "__main__":
    main()
