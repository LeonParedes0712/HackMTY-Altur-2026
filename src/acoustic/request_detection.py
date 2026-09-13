"""Envía un WAV al servicio local sin dependencias adicionales."""
import argparse
import base64
import json
from pathlib import Path
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--url", default="http://127.0.0.1:8000/detect")
    args = parser.parse_args()
    body = json.dumps({"audio": base64.b64encode(args.audio.read_bytes()).decode("ascii")}).encode()
    request = Request(args.url, data=body, headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=600) as response:
        print(json.dumps(json.load(response), indent=2))


if __name__ == "__main__":
    main()
