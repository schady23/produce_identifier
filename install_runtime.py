"""Install the Windows prototype runtime in .venv and fetch official model weights."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import urllib.request
import venv

ROOT=Path(__file__).resolve().parent
MODELS={
    "yolo11n.pt":("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                  "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"),
    "yolov8n-oiv7.pt":("https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n-oiv7.pt",
                       "3851dfbf39ed2a076b1f39215cc22dba64eb5646f282bf964703785b0bed6a41")}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline",action="store_true",help="install from cached .tools/wheels")
    parser.add_argument("--include-broad-model",action="store_true")
    args=parser.parse_args()
    if sys.version_info[:2] not in [(3,10),(3,11),(3,12)]:
        raise SystemExit("Use Python 3.10, 3.11, or 3.12 for these pinned dependencies.")
    interpreter=ROOT / ".venv/Scripts/python.exe"
    if not interpreter.exists():
        venv.EnvBuilder(with_pip=True).create(ROOT / ".venv")
    command=[str(interpreter),"-m","pip","--disable-pip-version-check","install"]
    if args.offline:
        command += ["--no-index","--find-links",str(ROOT / ".tools/wheels")]
    command += ["-r",str(ROOT / "requirements.txt")]
    subprocess.run(command,cwd=ROOT,check=True)
    (ROOT / "models").mkdir(exist_ok=True)
    for name in ["yolo11n.pt"]+(["yolov8n-oiv7.pt"] if args.include_broad_model else []):
        target=ROOT / "models" / name
        url,digest=MODELS[name]
        if not target.exists():
            print("Downloading official model:",name)
            temporary=target.with_suffix(".download")
            urllib.request.urlretrieve(url,temporary)
            if hashlib.sha256(temporary.read_bytes()).hexdigest()!=digest:
                raise RuntimeError("Model checksum mismatch: "+name)
            temporary.replace(target)
        if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
            raise RuntimeError("Unexpected model revision: "+name)
    print("Setup complete. Double-click start.cmd, or run .venv/Scripts/python.exe live_app.py")

if __name__=="__main__":
    main()
