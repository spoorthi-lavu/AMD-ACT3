#!/usr/bin/env python3
"""Review Forensics launcher: run `python run.py` (needs Python 3.9+; no installs). Opens the web app in your browser."""
import sys, os, threading, webbrowser
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
