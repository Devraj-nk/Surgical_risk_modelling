from __future__ import annotations

import json
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    # ui/server.py -> project root
    return Path(__file__).resolve().parents[1]


PROJECT_ROOT = _project_root()
STATIC_DIR = PROJECT_ROOT / "ui" / "static"
MODEL_PATH_DEFAULT = PROJECT_ROOT / "baseline_logreg_ovr.joblib"

# Reuse the exact schema used by the CLI to avoid drift.
sys.path.insert(0, str(PROJECT_ROOT))
from predict_risk import FEATURE_COLS, TARGET_COLS  # noqa: E402


def _json_response(handler: BaseHTTPRequestHandler, status: int, payload: dict[str, Any]) -> None:
    body = (json.dumps(payload, indent=2) + "\n").encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Cache-Control", "no-store")
    handler.end_headers()
    handler.wfile.write(body)


def _read_json(handler: BaseHTTPRequestHandler) -> Any:
    length_raw = handler.headers.get("Content-Length", "0")
    try:
        length = int(length_raw)
    except ValueError as e:
        raise ValueError(f"Invalid Content-Length: {length_raw!r}") from e

    raw = handler.rfile.read(length) if length > 0 else b""
    try:
        return json.loads(raw.decode("utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError("Request body must be valid JSON") from e


def _coerce_float(value: Any, *, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid numeric value for '{name}': {value!r}") from e


class App:
    def __init__(self, model_path: Path) -> None:
        self.model_path = model_path
        self._model = None

    def load_model(self):
        if self._model is not None:
            return self._model

        from joblib import load

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self._model = load(self.model_path)
        return self._model

    def predict(self, features: dict[str, float]) -> dict[str, float]:
        import pandas as pd

        X = pd.DataFrame([features], columns=FEATURE_COLS)
        model = self.load_model()

        proba = model.predict_proba(X)
        probs = proba[0] if hasattr(proba, "shape") and proba.shape[0] == 1 else list(proba)[0]

        if len(probs) != len(TARGET_COLS):
            raise RuntimeError(
                f"Unexpected predict_proba output: got {len(probs)} probabilities for {len(TARGET_COLS)} targets."
            )

        return {t: float(p) for t, p in zip(TARGET_COLS, probs, strict=True)}


APP = App(MODEL_PATH_DEFAULT)


class Handler(BaseHTTPRequestHandler):
    def _serve_static(self, path: str) -> None:
        # Map URL paths into ui/static
        if path in ("/", ""):
            rel = "index.html"
        else:
            rel = path.lstrip("/")

        target = (STATIC_DIR / rel).resolve()
        if not str(target).startswith(str(STATIC_DIR.resolve())):
            self.send_error(404)
            return

        if not target.exists() or not target.is_file():
            self.send_error(404)
            return

        # Minimal content-types
        suffix = target.suffix.lower()
        ctype = {
            ".html": "text/html; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".json": "application/json; charset=utf-8",
            ".glb": "model/gltf-binary",
            ".ico": "image/x-icon",
        }.get(suffix, "application/octet-stream")

        body = target.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.startswith("/api/schema"):
            _json_response(
                self,
                200,
                {
                    "feature_cols": FEATURE_COLS,
                    "target_cols": TARGET_COLS,
                },
            )
            return

        self._serve_static(self.path)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/predict":
            self.send_error(404)
            return

        try:
            payload = _read_json(self)

            if isinstance(payload, dict) and "features" in payload and isinstance(payload["features"], dict):
                payload = payload["features"]

            if not isinstance(payload, dict):
                raise ValueError("JSON must be an object of {feature_name: value} or {features: {...}}.")

            missing = [c for c in FEATURE_COLS if c not in payload]
            if missing:
                _json_response(self, 400, {"error": "missing_features", "missing": missing})
                return

            features = {c: _coerce_float(payload[c], name=c) for c in FEATURE_COLS}
            probs = APP.predict(features)

            _json_response(
                self,
                200,
                {
                    "features": features,
                    "risk_probabilities": probs,
                    "driving_target": "respiratory_compromise_next_30s",
                    "driving_probability": float(probs.get("respiratory_compromise_next_30s", 0.0)),
                },
            )
        except FileNotFoundError as e:
            _json_response(self, 500, {"error": "model_not_found", "message": str(e)})
        except Exception as e:  # keep simple for local prototype
            _json_response(self, 400, {"error": "bad_request", "message": str(e)})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Keep terminal output minimal.
        return


def main() -> None:
    host = "127.0.0.1"
    port = 8000

    if not STATIC_DIR.exists():
        raise FileNotFoundError(f"Static dir not found: {STATIC_DIR}")

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"Serving UI on http://{host}:{port}")
    print(f"Model: {APP.model_path}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
