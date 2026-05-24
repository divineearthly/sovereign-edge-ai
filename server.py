import ctypes, os, json, subprocess
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

HOME = Path(os.environ["HOME"])
LIB = HOME / "libvedic_full.so"
LLAMA = HOME / "llama.cpp" / "build" / "bin" / "llama-cli"
MODEL = HOME / "vedic-slm-1.1b-q4.gguf"
CACHE = HOME / "offline_cache.json"

try:
    _lib = ctypes.CDLL(str(LIB))
    VEDIC = True
except:
    VEDIC = False

class H(BaseHTTPRequestHandler):
    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html):
        body = html.encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        p = urlparse(self.path).path
        q = parse_qs(urlparse(self.path).query)
        if p == "/health":
            self._json({"status": "ok", "vedic": VEDIC, "model": MODEL.exists(), "llama": LLAMA.exists(), "cache": CACHE.exists()})
        elif p == "/farm":
            s = q.get("season", ["kharif"])[0]
            try:
                c = json.load(open(CACHE))
                d = c.get("seasons", {}).get(s, {})
                m = datetime.utcnow().month
                w = c.get("monthly_weather", {}).get(str(m), {})
                self._json({"season": s, "month": m, "weather": w, "crops": d.get("crops", []), "pests": d.get("pest_alerts", []), "soil": d.get("soil_advice", "")})
            except Exception as e:
                self._json({"error": str(e)}, 500)
        elif p == "/cache":
            try:
                c = json.load(open(CACHE))
                self._json({"status": "available", "last_sync": c.get("last_sync", "never")})
            except:
                self._json({"status": "missing"})
        elif p == "/":
            self._html(open(HOME / "index.html").read() if (HOME / "index.html").exists() else "<h1>Sovereign Edge AI</h1><p>API running.</p>")
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        p = urlparse(self.path).path
        if p == "/infer":
            d = self._body()
            prompt = d.get("prompt", "")
            lang = d.get("lang", "en")
            if not prompt:
                self._json({"error": "prompt required"}, 400)
                return
            if not MODEL.exists() or not LLAMA.exists():
                self._json({"text": "[Offline] Model not available.", "cached": True})
                return
            sp = {"hi": "आप सहायक AI हैं। हिंदी में उत्तर दें।", "bn": "আপনি সহায়ক AI। বাংলায় উত্তর দিন।", "as": "আপুনি সহায়ক AI। অসমীয়াত উত্তৰ দিয়ক।", "ta": "நீங்கள் உதவி AI. தமிழில் பதிலளிக்கவும்.", "en": "You are a helpful AI assistant."}
            full = sp.get(lang, sp["en"]) + "\nUser: " + prompt + "\nAssistant:"
            try:
                r = subprocess.run([str(LLAMA), "-m", str(MODEL), "-p", full, "-n", str(d.get("max_tokens", 128)), "--temp", str(d.get("temperature", 0.7)), "--no-display-prompt", "-t", "4"], capture_output=True, text=True, timeout=300)
                self._json({"text": r.stdout.strip(), "cached": False})
            except subprocess.TimeoutExpired:
                self._json({"text": "[Timeout]", "cached": True})
            except Exception as e:
                self._json({"text": "[Error] " + str(e), "cached": True})
        elif p == "/kavach":
            d = self._body()
            t = d.get("text", "").lower()
            threats = ["otp", "account blocked", "click link", "verify now", "bank account", "password", "suspended", "send money"]
            safe = ["paddy harvest", "rain forecast", "market price", "soil health", "crop rotation", "organic farming"]
            ft = [x for x in threats if x in t]
            fs = [x for x in safe if x in t]
            score = len(ft) - len(fs) * 0.5
            v = "THREAT" if score > 1 else ("SUSPICIOUS" if score > 0 else "SAFE")
            self._json({"verdict": v, "score": score, "threats": ft, "safe": fs})
        else:
            self._json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

if __name__ == "__main__":
    print("Sovereign Edge AI Server")
    print("Vedic backend:", "LOADED" if VEDIC else "MISSING")
    print("http://0.0.0.0:8000")
    HTTPServer(("0.0.0.0", 8000), H).serve_forever()
