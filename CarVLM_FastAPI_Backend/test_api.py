import sys, requests
from pathlib import Path
if len(sys.argv) != 2:
    raise SystemExit("Usage: python test_api.py path/to/car.jpg")
path = sys.argv[1]
with open(path, "rb") as f:
    r = requests.post("http://127.0.0.1:8000/predict",
                      files={"file": (Path(path).name, f, "application/octet-stream")},
                      timeout=180)
print(r.status_code)
print(r.text)
