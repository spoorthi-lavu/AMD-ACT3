#!/usr/bin/env python3
"""Review Forensics launcher: run `python run.py` (needs Python 3.9+; no installs). Opens the web app in your browser."""
import sys, os, threading, webbrowser

# Load local .env without external dependencies
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.isfile(_env_path):
    with open(_env_path, encoding="utf-8") as _f:
        for _l in _f:
            _l = _l.strip()
            if _l and not _l.startswith("#") and "=" in _l:
                _k, _v = _l.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip().strip('"').strip("'"))

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend"))
from app.server import serve
port = int(os.environ.get("PORT", 8765))
try: srv = serve(port)
except OSError: sys.exit(f"Port {port} is busy. Run with another port, e.g.  PORT=9000 python run.py")
url = f"http://127.0.0.1:{port}/"
print(f"Review Forensics running at {url}   (Ctrl+C to stop)")
threading.Timer(1, lambda: webbrowser.open(url)).start()
try: srv.serve_forever()
except KeyboardInterrupt: print("\nStopped.")
