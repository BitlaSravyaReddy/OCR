import argparse
import contextlib
import csv
import html
import http.server
import json
import os
import queue
import shutil
import socketserver
import sys
import threading
import time
import tkinter as tk
import webbrowser
import winreg
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional

# Keep Paddle from doing slow network host checks when the engine is eventually loaded.
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
os.environ.setdefault("PADDLE_PDX_EAGER_INIT", "False")
os.environ.setdefault("FLAGS_use_mkldnn", "0")


def _configure_windows_dll_search_paths() -> None:
    if os.name != "nt":
        return

    candidate_dirs: List[Path] = []
    base_dir = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

    candidate_dirs.extend(
        [
            base_dir,
            base_dir / "paddle" / "libs",
            base_dir / "paddle" / "base",
            base_dir / "paddle" / "base" / ".." / "libs",
            base_dir / "torch" / "lib",
            base_dir / "cv2",
        ]
    )

    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidate_dirs.extend(
            [
                exe_dir,
                exe_dir / "_internal",
                exe_dir / "_internal" / "paddle" / "libs",
                exe_dir / "_internal" / "paddle" / "base",
                exe_dir / "_internal" / "torch" / "lib",
                exe_dir / "_internal" / "cv2",
            ]
        )

    prefix_root = Path(sys.prefix)
    candidate_dirs.extend(
        [
            prefix_root / "Lib" / "site-packages" / "paddle",
            prefix_root / "Lib" / "site-packages" / "paddle" / "libs",
            prefix_root / "Lib" / "site-packages" / "paddle" / "base",
            prefix_root / "Lib" / "site-packages" / "torch" / "lib",
            prefix_root / "Lib" / "site-packages" / "cv2",
        ]
    )

    seen: set[str] = set()
    valid_dirs: List[str] = []
    for directory in candidate_dirs:
        try:
            resolved = str(directory.resolve())
        except Exception:
            resolved = str(directory)
        if resolved in seen:
            continue
        seen.add(resolved)
        if os.path.isdir(resolved):
            valid_dirs.append(resolved)

    if valid_dirs:
        os.environ["PATH"] = os.pathsep.join(valid_dirs + [os.environ.get("PATH", "")])
        add_dll_directory = getattr(os, "add_dll_directory", None)
        if callable(add_dll_directory):
            for directory in valid_dirs:
                try:
                    add_dll_directory(directory)
                except Exception:
                    pass


_configure_windows_dll_search_paths()

from PIL import Image, ImageDraw

try:
    import pystray
except Exception:  # pragma: no cover
    pystray = None


APP_NAME = "InvoiceBatchProcessor"
CONFIG_DIR = Path(os.getenv("APPDATA", str(Path.home()))) / APP_NAME
CONFIG_PATH = CONFIG_DIR / "watcher_config.json"
SYSTEM_FOLDER_NAME = "_INVOICE_SYSTEM"
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
REQUIRED_PADDLE_MODEL_DIRS = ("PP-OCRv5_server_det", "en_PP-OCRv5_mobile_rec")
_BATCH_APP: Any = None
_BATCH_APP_LOAD_ERROR: Optional[str] = None
_BATCH_APP_LOCK = threading.Lock()


class _QueueWriter:
    def __init__(self, sink: queue.Queue[str]) -> None:
        self._sink = sink

    def write(self, data: str) -> None:
        if data and data.strip():
            self._sink.put(data.rstrip("\n"))

    def flush(self) -> None:
        return


def _default_config() -> Dict[str, Any]:
    return {
        "database_root": "",
        "poll_seconds": 120,
        "startup_enabled": True,
        "openai_api_key": "",
        "google_api_key": "",
    }


def _load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return _default_config()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        merged = _default_config()
        if isinstance(data, dict):
            merged.update(data)
        return merged
    except Exception:
        return _default_config()


def _save_config(config: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")


def _apply_runtime_credentials(config: Dict[str, Any]) -> None:
    openai_key = str(config.get("openai_api_key") or "").strip()
    google_key = str(config.get("google_api_key") or "").strip()

    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key
    if google_key:
        os.environ["GOOGLE_API_KEY"] = google_key
        os.environ["GEMINI_API_KEY"] = google_key


FOLDER_KEYWORDS = {
    "purchase": ("purchase", "purchases", "purch"),
    "sales": ("sales", "sale"),
    "completed": ("completed", "complete", "done", "processed"),
    "json": ("json",),
}


def _folder_matches(name: str, target_name: str) -> bool:
    normalized = name.lower().replace("_", " ").replace("-", " ").strip()
    keywords = FOLDER_KEYWORDS.get(target_name.lower(), (target_name.lower(),))
    return any(keyword in normalized for keyword in keywords)


def _get_caseflex_subdir(parent: Path, target_name: str) -> Optional[Path]:
    for child in parent.iterdir():
        if child.is_dir() and _folder_matches(child.name, target_name):
            return child
    return None


def _find_caseflex_subdir(parent: Path, target_name: str, create_name: str) -> Path:
    for child in parent.iterdir():
        if child.is_dir() and _folder_matches(child.name, target_name):
            return child
    path = parent / create_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def _is_system_folder(path: Path) -> bool:
    return path.name.lower() == SYSTEM_FOLDER_NAME.lower()


def _sanitize_client_name(name: str) -> str:
    cleaned = "".join(ch for ch in name.strip() if ch not in '<>:"/\\|?*')
    cleaned = " ".join(cleaned.split())
    return cleaned


def _app_base_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _ensure_paddle_model_cache(log: Optional[Any] = None) -> None:
    source_candidates = [
        _app_base_dir() / "paddle_models",
        Path(__file__).resolve().parent / "release_assets" / "paddle_models",
    ]
    source_root = next((path for path in source_candidates if path.exists()), None)
    if source_root is None:
        if log:
            log("PADDLE MODEL CACHE | bundled models not found; Paddle will use/download its normal cache.")
        return

    target_root = Path.home() / ".paddlex" / "official_models"
    target_root.mkdir(parents=True, exist_ok=True)
    for model_dir_name in REQUIRED_PADDLE_MODEL_DIRS:
        source_dir = source_root / model_dir_name
        target_dir = target_root / model_dir_name
        if not source_dir.exists():
            continue
        if (target_dir / "inference.pdiparams").exists():
            if log:
                log(f"PADDLE MODEL CACHE HIT | {model_dir_name}")
            continue
        if log:
            log(f"PADDLE MODEL CACHE COPY | {model_dir_name} -> {target_dir}")
        if target_dir.exists():
            shutil.rmtree(target_dir, ignore_errors=True)
        shutil.copytree(source_dir, target_dir)


def _load_batch_app(log: Optional[Any] = None) -> Any:
    global _BATCH_APP, _BATCH_APP_LOAD_ERROR
    if _BATCH_APP is not None:
        return _BATCH_APP
    with _BATCH_APP_LOCK:
        if _BATCH_APP is not None:
            return _BATCH_APP
        if _BATCH_APP_LOAD_ERROR:
            raise RuntimeError(
                "Engine initialization failed earlier in this app session. "
                "Please restart the application before trying again. "
                f"Last error: {_BATCH_APP_LOAD_ERROR}"
            )
        started = time.perf_counter()
        if log:
            log("ENGINE LOAD START | loading OCR/LLM modules. First load can take a little time.")
        _ensure_paddle_model_cache(log)
        try:
            import app as loaded_batch_app  # Imported lazily so the UI opens before heavy OCR modules load.
        except Exception as exc:
            _BATCH_APP_LOAD_ERROR = str(exc)
            if log:
                log(
                    "ENGINE LOAD FAILED | "
                    "restart the desktop app before retrying to avoid reinitializing PaddleX. "
                    f"error={exc}"
                )
            raise

        _BATCH_APP = loaded_batch_app
        _BATCH_APP_LOAD_ERROR = None
        if log:
            log(f"ENGINE LOAD DONE | duration_ms={int((time.perf_counter() - started) * 1000)}")
        return _BATCH_APP


def _resolve_unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    candidate = path.with_name(f"{path.stem}_{stamp}{path.suffix}")
    idx = 1
    while candidate.exists():
        candidate = path.with_name(f"{path.stem}_{stamp}_{idx}{path.suffix}")
        idx += 1
    return candidate


def _is_supported_input(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.name.startswith("~$")


def _is_file_ready(path: Path, min_age_seconds: int = 15) -> bool:
    try:
        age = time.time() - path.stat().st_mtime
        if age < min_age_seconds:
            return False
        with path.open("rb"):
            return True
    except Exception:
        return False


def _startup_command() -> str:
    if getattr(sys, "frozen", False):
        return f'"{sys.executable}" --start-minimized'
    return f'"{sys.executable}" "{Path(__file__).resolve()}" --start-minimized'


def _startup_script_path() -> Path:
    appdata = Path(os.getenv("APPDATA", str(Path.home())))
    return appdata / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup" / f"{APP_NAME}.cmd"


def _set_startup_script_enabled(enabled: bool) -> None:
    script_path = _startup_script_path()
    try:
        if enabled:
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text(
                "@echo off\n"
                f"start \"\" {_startup_command()}\n",
                encoding="utf-8",
            )
        elif script_path.exists():
            script_path.unlink()
    except Exception:
        return


def _set_startup_enabled(enabled: bool) -> None:
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Run",
            0,
            winreg.KEY_SET_VALUE,
        )
        with key:
            if enabled:
                winreg.SetValueEx(key, APP_NAME, 0, winreg.REG_SZ, _startup_command())
            else:
                try:
                    winreg.DeleteValue(key, APP_NAME)
                except FileNotFoundError:
                    pass
    except Exception:
        pass
    _set_startup_script_enabled(enabled)


class _ThreadingHTTPServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class _DashboardRequestHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        watcher_app = getattr(self.server, "watcher_app", None)
        if watcher_app is None:
            self.send_error(500, "Watcher app not attached")
            return

        if self.path in ("/", "/index.html"):
            payload = watcher_app.render_dashboard_html()
            body = payload.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/latest":
            body = json.dumps(watcher_app.get_dashboard_state(), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        self.send_error(404, "Not Found")

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


class StartupSplash:
    def __init__(self, root: tk.Tk) -> None:
        self.window = tk.Toplevel(root)
        self.window.title("Starting Invoice Batch Processor")
        self.window.geometry("520x180")
        self.window.resizable(False, False)
        self.window.configure(bg="#f8fafc")
        self.window.protocol("WM_DELETE_WINDOW", lambda: None)

        frame = ttk.Frame(self.window, padding=18)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="Invoice Batch Processor", font=("Segoe UI", 15, "bold")).pack(anchor="w")
        ttk.Label(
            frame,
            text="Preparing watcher, dashboard, and tray controls...",
            foreground="#475569",
        ).pack(anchor="w", pady=(8, 12))
        self.status = tk.StringVar(value="Starting application...")
        ttk.Label(frame, textvariable=self.status).pack(anchor="w", pady=(0, 10))
        self.progress = ttk.Progressbar(frame, mode="indeterminate")
        self.progress.pack(fill="x")
        self.progress.start(12)
        self.window.update_idletasks()
        self._center()

    def _center(self) -> None:
        self.window.update_idletasks()
        width = self.window.winfo_width()
        height = self.window.winfo_height()
        x = (self.window.winfo_screenwidth() // 2) - (width // 2)
        y = (self.window.winfo_screenheight() // 2) - (height // 2)
        self.window.geometry(f"{width}x{height}+{x}+{y}")

    def set_status(self, text: str) -> None:
        self.status.set(text)
        self.window.update_idletasks()

    def close(self) -> None:
        try:
            self.progress.stop()
            self.window.destroy()
        except Exception:
            pass


def _render_admin_html(summary: Dict[str, Any], rows: List[Dict[str, Any]]) -> str:
    total = len(rows)
    success = sum(1 for row in rows if row.get("status") == "success")
    failed = total - success
    llm_success = sum(1 for row in rows if row.get("llm_status") == "success")
    llm_fallback = sum(1 for row in rows if row.get("llm_status") == "fallback")

    def _card(label: str, value: Any) -> str:
        return (
            "<div class='card'>"
            f"<div class='label'>{html.escape(str(label))}</div>"
            f"<div class='value'>{html.escape(str(value))}</div>"
            "</div>"
        )

    rows_html: List[str] = []
    for row in rows:
        admin_fields = row.get("admin_fields", {}) if isinstance(row.get("admin_fields"), dict) else {}
        rows_html.append(
            "<tr>"
            f"<td>{html.escape(str(row.get('client_name') or ''))}</td>"
            f"<td>{html.escape(str(row.get('invoice_type') or ''))}</td>"
            f"<td>{html.escape(str(Path(str(row.get('file') or '')).name))}</td>"
            f"<td>{html.escape(str(row.get('status') or ''))}</td>"
            f"<td>{html.escape(str(row.get('llm_status') or ''))}</td>"
            f"<td>{html.escape(str(admin_fields.get('invoice_number') or ''))}</td>"
            f"<td>{html.escape(str(admin_fields.get('supplier_name') or ''))}</td>"
            f"<td>{html.escape(str(admin_fields.get('customer_name') or ''))}</td>"
            f"<td>{html.escape(str(admin_fields.get('total_amount') or ''))}</td>"
            f"<td>{html.escape(str(row.get('duration_ms') or ''))}</td>"
            f"<td>{html.escape(str(row.get('output_json') or ''))}</td>"
            "</tr>"
        )

    generated_at = html.escape(str(summary.get("processed_at") or ""))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Invoice Watcher Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f4f7fb; color: #1f2937; }}
    .muted {{ color: #6b7280; margin-bottom: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .card {{ background: white; border-radius: 12px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 12px; overflow: hidden; }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; }}
    th {{ background: #0f172a; color: white; }}
    tr:nth-child(even) td {{ background: #f8fafc; }}
  </style>
</head>
<body>
  <h1>Invoice Watcher Dashboard</h1>
  <div class="muted">Generated at: {generated_at}</div>
  <div class="grid">
    {_card("Processed", total)}
    {_card("Success", success)}
    {_card("Failed", failed)}
    {_card("LLM Success", llm_success)}
    {_card("LLM Fallback", llm_fallback)}
  </div>
  <table>
    <thead>
      <tr>
        <th>Client</th>
        <th>Type</th>
        <th>File</th>
        <th>Status</th>
        <th>LLM</th>
        <th>Invoice No</th>
        <th>Supplier</th>
        <th>Customer</th>
        <th>Total</th>
        <th>Duration (ms)</th>
        <th>Output JSON</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows_html)}
    </tbody>
  </table>
</body>
</html>
"""


class WatcherDesktopApp:
    def __init__(self, root: tk.Tk, start_minimized: bool = False) -> None:
        self.root = root
        self.root.title("Invoice Database Watcher")
        self.root.geometry("1080x760")

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.stop_event = threading.Event()
        self.monitor_thread: Optional[threading.Thread] = None
        self.tray_icon: Any = None
        self.client_index_map: List[str] = []
        self.dashboard_server: Optional[_ThreadingHTTPServer] = None
        self.dashboard_thread: Optional[threading.Thread] = None
        self.dashboard_port: Optional[int] = None

        self.config = _load_config()
        _apply_runtime_credentials(self.config)
        self.database_root = tk.StringVar(value=str(self.config.get("database_root") or ""))
        self.poll_seconds = tk.StringVar(value=str(self.config.get("poll_seconds") or 120))
        self.startup_enabled = tk.BooleanVar(value=bool(self.config.get("startup_enabled")))
        self.openai_api_key = tk.StringVar(value=str(self.config.get("openai_api_key") or ""))
        self.google_api_key = tk.StringVar(value=str(self.config.get("google_api_key") or ""))
        self.new_client_name = tk.StringVar(value="")

        self._build_ui()
        self._start_dashboard_server()
        self._create_tray_icon()
        self._poll_logs()
        self.root.protocol("WM_DELETE_WINDOW", self._handle_window_close)
        self._refresh_client_list()

        if self.database_root.get().strip():
            self._start_monitoring()

        if start_minimized and pystray is not None:
            self.root.after(300, self.hide_to_tray)

    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 6}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=10, pady=10)

        ttk.Label(frm, text="Database Folder").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.database_root).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(frm, text="Browse", command=self._pick_database_root).grid(row=0, column=2, sticky="ew", **pad)

        ttk.Label(frm, text="Poll Every (seconds)").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.poll_seconds).grid(row=1, column=1, sticky="ew", **pad)

        ttk.Label(frm, text="OpenAI API Key").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.openai_api_key, show="*").grid(row=2, column=1, sticky="ew", **pad)

        ttk.Label(frm, text="Google AI Key (optional fallback)").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(frm, textvariable=self.google_api_key, show="*").grid(row=3, column=1, sticky="ew", **pad)

        ttk.Checkbutton(frm, text="Start with Windows", variable=self.startup_enabled).grid(row=4, column=0, sticky="w", **pad)

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=5, column=0, columnspan=3, sticky="ew", padx=8, pady=8)
        ttk.Button(btn_row, text="Save Settings", command=self._save_settings).pack(side="left", padx=4)
        self.start_btn = ttk.Button(btn_row, text="Start Monitoring", command=self._toggle_monitoring)
        self.start_btn.pack(side="left", padx=4)
        ttk.Button(btn_row, text="Open Dashboard", command=self.open_dashboard).pack(side="left", padx=4)
        ttk.Button(btn_row, text="Hide to Tray", command=self.hide_to_tray).pack(side="left", padx=4)

        client_frame = ttk.LabelFrame(frm, text="Client Folder Structure")
        client_frame.grid(row=6, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)
        client_frame.columnconfigure(0, weight=1)

        self.client_list = tk.Listbox(client_frame, height=10)
        self.client_list.grid(row=0, column=0, columnspan=3, sticky="nsew", padx=8, pady=8)

        ttk.Label(client_frame, text="New Client Name").grid(row=1, column=0, sticky="w", padx=8, pady=4)
        ttk.Entry(client_frame, textvariable=self.new_client_name).grid(row=1, column=1, sticky="ew", padx=8, pady=4)
        ttk.Button(client_frame, text="Add New Client", command=self._add_new_client).grid(row=1, column=2, sticky="ew", padx=8, pady=4)

        ttk.Button(client_frame, text="Refresh Clients", command=self._refresh_client_list).grid(row=2, column=0, sticky="ew", padx=8, pady=6)
        ttk.Label(
            client_frame,
            text="Structure: Client / Purchase / (Completed, JSON) and Client / Sales / (Completed, JSON). Existing folder names are matched by keywords.",
        ).grid(row=2, column=1, columnspan=2, sticky="w", padx=8, pady=6)
        client_frame.rowconfigure(0, weight=1)

        ttk.Label(frm, text="Live Monitor Log").grid(row=7, column=0, sticky="w", **pad)
        self.log_text = tk.Text(frm, height=20, wrap="word")
        self.log_text.grid(row=8, column=0, columnspan=3, sticky="nsew", padx=8, pady=6)
        scroll = ttk.Scrollbar(frm, command=self.log_text.yview)
        scroll.grid(row=8, column=3, sticky="ns")
        self.log_text.configure(yscrollcommand=scroll.set)

        frm.columnconfigure(1, weight=1)
        frm.rowconfigure(8, weight=1)

    def _get_database_root(self) -> Optional[Path]:
        raw = self.database_root.get().strip()
        if not raw:
            return None
        return Path(raw)

    def _get_system_root(self) -> Optional[Path]:
        root = self._get_database_root()
        if root is None:
            return None
        return root / SYSTEM_FOLDER_NAME

    def _start_dashboard_server(self) -> None:
        if self.dashboard_server is not None:
            return
        server: Optional[_ThreadingHTTPServer] = None
        for port in range(8765, 8776):
            try:
                server = _ThreadingHTTPServer(("127.0.0.1", port), _DashboardRequestHandler)
                break
            except OSError:
                continue
        if server is None:
            self.log_queue.put("DASHBOARD FAILED | no free local port between 8765 and 8775")
            return
        server.watcher_app = self  # type: ignore[attr-defined]
        self.dashboard_server = server
        self.dashboard_port = int(server.server_address[1])
        self.dashboard_thread = threading.Thread(target=server.serve_forever, daemon=True)
        self.dashboard_thread.start()
        self.log_queue.put(f"DASHBOARD READY | url={self.dashboard_url}")

    @property
    def dashboard_url(self) -> str:
        port = self.dashboard_port or 8765
        return f"http://127.0.0.1:{port}/"

    def open_dashboard(self) -> None:
        if self.dashboard_server is None:
            self._start_dashboard_server()
        webbrowser.open(self.dashboard_url)

    def get_dashboard_state(self) -> Dict[str, Any]:
        root = self._get_database_root()
        system_root = self._get_system_root()
        latest_scan: Dict[str, Any] = {}
        recent_rows: List[Dict[str, Any]] = []
        history_rows: List[Dict[str, Any]] = []
        errors: List[Dict[str, Any]] = []

        if system_root is not None:
            admin_root = system_root / "admin"
            latest_scan_path = admin_root / "latest_scan.json"
            history_csv_path = admin_root / "history.csv"
            if latest_scan_path.exists():
                try:
                    latest_scan = json.loads(latest_scan_path.read_text(encoding="utf-8"))
                except Exception:
                    latest_scan = {}
            if history_csv_path.exists():
                try:
                    with history_csv_path.open("r", encoding="utf-8", newline="") as fh:
                        history_rows = list(csv.DictReader(fh))
                except Exception:
                    history_rows = []

        if history_rows:
            recent_rows = history_rows[-50:][::-1]
            errors = [row for row in recent_rows if str(row.get("status") or "").lower() != "success"][:20]

        monitoring_active = bool(self.monitor_thread and self.monitor_thread.is_alive())
        client_count = 0
        if root is not None and root.exists():
            try:
                client_count = len([p for p in root.iterdir() if p.is_dir() and not _is_system_folder(p)])
            except Exception:
                client_count = 0

        return {
            "app_name": APP_NAME,
            "dashboard_url": self.dashboard_url,
            "monitoring_active": monitoring_active,
            "database_root": str(root) if root else "",
            "poll_seconds": int(self.config.get("poll_seconds", 120) or 120),
            "client_count": client_count,
            "latest_scan": latest_scan,
            "recent_rows": recent_rows,
            "recent_errors": errors,
            "generated_at": datetime.now().isoformat(timespec="seconds"),
        }

    def render_dashboard_html(self) -> str:
        state = self.get_dashboard_state()
        latest_scan = state.get("latest_scan", {}) if isinstance(state.get("latest_scan"), dict) else {}
        recent_rows = state.get("recent_rows", []) if isinstance(state.get("recent_rows"), list) else []
        recent_errors = state.get("recent_errors", []) if isinstance(state.get("recent_errors"), list) else []
        monitoring_active = bool(state.get("monitoring_active"))
        status_text = "Running" if monitoring_active else "Stopped"
        status_color = "#16a34a" if monitoring_active else "#dc2626"

        def _card(label: str, value: Any) -> str:
            return (
                "<div class='card'>"
                f"<div class='label'>{html.escape(str(label))}</div>"
                f"<div class='value'>{html.escape(str(value))}</div>"
                "</div>"
            )

        row_html: List[str] = []
        for row in recent_rows[:25]:
            row_html.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('processed_at') or ''))}</td>"
                f"<td>{html.escape(str(row.get('client_name') or ''))}</td>"
                f"<td>{html.escape(str(row.get('bucket') or ''))}</td>"
                f"<td>{html.escape(str(row.get('invoice_type') or ''))}</td>"
                f"<td>{html.escape(str(Path(str(row.get('file') or '')).name))}</td>"
                f"<td>{html.escape(str(row.get('status') or ''))}</td>"
                f"<td>{html.escape(str(row.get('llm_status') or ''))}</td>"
                f"<td>{html.escape(str(row.get('invoice_number') or ''))}</td>"
                f"<td>{html.escape(str(row.get('supplier_name') or ''))}</td>"
                f"<td>{html.escape(str(row.get('customer_name') or ''))}</td>"
                f"<td>{html.escape(str(row.get('total_amount') or ''))}</td>"
                f"<td>{html.escape(str(row.get('error') or ''))}</td>"
                "</tr>"
            )

        error_html: List[str] = []
        for row in recent_errors:
            error_html.append(
                "<tr>"
                f"<td>{html.escape(str(row.get('processed_at') or ''))}</td>"
                f"<td>{html.escape(str(row.get('client_name') or ''))}</td>"
                f"<td>{html.escape(str(row.get('bucket') or ''))}</td>"
                f"<td>{html.escape(str(Path(str(row.get('file') or '')).name))}</td>"
                f"<td>{html.escape(str(row.get('error') or ''))}</td>"
                "</tr>"
            )

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta http-equiv="refresh" content="10">
  <title>Invoice Watcher Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background: #f5f7fb; color: #111827; }}
    .top {{ display: flex; justify-content: space-between; align-items: center; gap: 16px; margin-bottom: 18px; }}
    .title p {{ color: #6b7280; margin: 6px 0 0; }}
    .status {{ display: inline-flex; align-items: center; gap: 8px; font-weight: 600; background: white; padding: 10px 14px; border-radius: 999px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .dot {{ width: 10px; height: 10px; border-radius: 50%; background: {status_color}; }}
    .grid {{ display: grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap: 12px; margin: 20px 0 28px; }}
    .card {{ background: white; border-radius: 14px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    .label {{ font-size: 12px; color: #6b7280; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 28px; font-weight: 700; margin-top: 8px; }}
    .section {{ margin-top: 26px; }}
    .meta {{ background: white; border-radius: 14px; padding: 16px; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 14px; overflow: hidden; box-shadow: 0 2px 10px rgba(0,0,0,0.06); }}
    th, td {{ padding: 10px 12px; border-bottom: 1px solid #e5e7eb; text-align: left; font-size: 13px; vertical-align: top; }}
    th {{ background: #0f172a; color: white; }}
    tr:nth-child(even) td {{ background: #f8fafc; }}
    code {{ background: #eef2ff; padding: 2px 6px; border-radius: 6px; }}
  </style>
</head>
<body>
  <div class="top">
    <div class="title">
      <h1>Invoice Watcher Dashboard</h1>
      <p>Separate live browser dashboard for monitoring processing health and recent failures.</p>
    </div>
    <div class="status"><span class="dot"></span>{html.escape(status_text)}</div>
  </div>

    <div class="meta">
    <div><strong>Database Root:</strong> <code>{html.escape(str(state.get('database_root') or 'Not configured'))}</code></div>
    <div><strong>Dashboard URL:</strong> <code>{html.escape(str(state.get('dashboard_url') or ''))}</code></div>
    <div><strong>Poll Every:</strong> {html.escape(str(state.get('poll_seconds') or ''))} seconds</div>
    <div><strong>Routing:</strong> Client / Purchase -> Purchase prompt, Client / Sales -> Sales prompt</div>
    <div><strong>Generated At:</strong> {html.escape(str(state.get('generated_at') or ''))}</div>
  </div>

  <div class="grid">
    {_card("Clients", state.get("client_count", 0))}
    {_card("Last Scan Files", latest_scan.get("files_processed", 0))}
    {_card("Last Scan Success", latest_scan.get("success", 0))}
    {_card("Last Scan Failed", latest_scan.get("failed", 0))}
    {_card("Duration (ms)", latest_scan.get("duration_ms", 0))}
    {_card("Recent Errors", len(recent_errors))}
  </div>

  <div class="section">
    <h2>Recent Errors</h2>
    <table>
      <thead>
        <tr>
          <th>Processed At</th>
          <th>Client</th>
          <th>Bucket</th>
          <th>File</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>
        {''.join(error_html) if error_html else '<tr><td colspan="5">No recent errors.</td></tr>'}
      </tbody>
    </table>
  </div>

  <div class="section">
    <h2>Recent Activity</h2>
    <table>
      <thead>
        <tr>
          <th>Processed At</th>
          <th>Client</th>
          <th>Bucket</th>
          <th>Type</th>
          <th>File</th>
          <th>Status</th>
          <th>LLM</th>
          <th>Invoice No</th>
          <th>Supplier</th>
          <th>Customer</th>
          <th>Total</th>
          <th>Error</th>
        </tr>
      </thead>
      <tbody>
        {''.join(row_html) if row_html else '<tr><td colspan="12">No processed files yet.</td></tr>'}
      </tbody>
    </table>
  </div>
</body>
</html>
"""

    def _pick_database_root(self) -> None:
        selected = filedialog.askdirectory()
        if selected:
            self.database_root.set(selected)
            self._refresh_client_list()

    def _refresh_client_list(self) -> None:
        self.client_list.delete(0, tk.END)
        self.client_index_map = []
        root = Path(self.database_root.get().strip()) if self.database_root.get().strip() else None
        if root is None or not root.exists():
            return
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            if _is_system_folder(child):
                continue
            self.client_index_map.append(child.name)
            purchase_dir = _get_caseflex_subdir(child, "purchase")
            sales_dir = _get_caseflex_subdir(child, "sales")
            purchase_queue = 0
            sales_queue = 0
            purchase_completed = None
            purchase_json = None
            sales_completed = None
            sales_json = None
            if purchase_dir and purchase_dir.exists():
                purchase_queue = len([p for p in purchase_dir.iterdir() if _is_supported_input(p)])
                purchase_completed = _get_caseflex_subdir(purchase_dir, "completed")
                purchase_json = _get_caseflex_subdir(purchase_dir, "json")
            if sales_dir and sales_dir.exists():
                sales_queue = len([p for p in sales_dir.iterdir() if _is_supported_input(p)])
                sales_completed = _get_caseflex_subdir(sales_dir, "completed")
                sales_json = _get_caseflex_subdir(sales_dir, "json")
            status = (
                f"{child.name} | "
                f"Purchase={'OK' if purchase_dir else 'Missing'} "
                f"(Completed={'OK' if purchase_completed else 'Missing'}, JSON={'OK' if purchase_json else 'Missing'}, queue={purchase_queue}) | "
                f"Sales={'OK' if sales_dir else 'Missing'} "
                f"(Completed={'OK' if sales_completed else 'Missing'}, JSON={'OK' if sales_json else 'Missing'}, queue={sales_queue})"
            )
            self.client_list.insert(tk.END, status)

    def _create_client_structure(self, client_dir: Path) -> None:
        for bucket_name in ("Purchase", "Sales"):
            bucket_dir = _find_caseflex_subdir(client_dir, bucket_name.lower(), bucket_name)
            _find_caseflex_subdir(bucket_dir, "completed", "Completed")
            _find_caseflex_subdir(bucket_dir, "json", "JSON")

    def _add_new_client(self) -> None:
        root_path = self.database_root.get().strip()
        if not root_path:
            messagebox.showerror("Missing Folder", "Please select the database folder first.")
            return
        root = Path(root_path)
        root.mkdir(parents=True, exist_ok=True)

        client_name = _sanitize_client_name(self.new_client_name.get())
        if not client_name:
            messagebox.showerror("Invalid Client", "Please enter a valid client name.")
            return

        existing_client = None
        for child in root.iterdir():
            if child.is_dir() and not _is_system_folder(child) and child.name.lower() == client_name.lower():
                existing_client = child
                break

        client_dir = existing_client or (root / client_name)
        client_dir.mkdir(parents=True, exist_ok=True)
        self._create_client_structure(client_dir)

        self.new_client_name.set("")
        self._save_settings()
        self._refresh_client_list()
        self._append_log(f"CLIENT READY | client={client_dir} | structure=Purchase/Sales/Completed/JSON")
        messagebox.showinfo("Client Ready", f"Folder structure created for:\n{client_dir}")

    def _save_settings(self) -> None:
        root_path = self.database_root.get().strip()
        if not root_path:
            messagebox.showerror("Missing Folder", "Database folder is required.")
            return
        try:
            poll_seconds = max(30, int(self.poll_seconds.get().strip()))
        except Exception:
            messagebox.showerror("Invalid Polling", "Poll seconds must be a valid integer >= 30.")
            return
        self.config["database_root"] = root_path
        self.config["poll_seconds"] = poll_seconds
        self.config["startup_enabled"] = bool(self.startup_enabled.get())
        self.config["openai_api_key"] = self.openai_api_key.get().strip()
        self.config["google_api_key"] = self.google_api_key.get().strip()
        _save_config(self.config)
        _apply_runtime_credentials(self.config)
        _set_startup_enabled(bool(self.startup_enabled.get()))
        self._refresh_client_list()
        self._append_log(f"SETTINGS SAVED | database_root={root_path} | poll_seconds={poll_seconds}")
        if self.config.get("openai_api_key"):
            self._append_log("OPENAI KEY CONFIGURED | primary LLM provider ready")
        else:
            self._append_log("OPENAI KEY MISSING | processing will fail unless key is available elsewhere")
        if self.config.get("google_api_key"):
            self._append_log("GOOGLE KEY CONFIGURED | Gemini fallback ready")
        else:
            self._append_log("GOOGLE KEY OPTIONAL | fallback disabled until key is provided")
        self._append_log(f"DASHBOARD URL | {self.dashboard_url}")

    def _toggle_monitoring(self) -> None:
        if self.monitor_thread and self.monitor_thread.is_alive():
            self._stop_monitoring()
        else:
            self._start_monitoring()

    def _start_monitoring(self) -> None:
        self._save_settings()
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
        self.stop_event.clear()
        self.monitor_thread = threading.Thread(target=self._watch_loop, daemon=True)
        self.monitor_thread.start()
        self.start_btn.config(text="Stop Monitoring")
        self._append_log("WATCHER STARTED")
        self._append_log(f"DASHBOARD LIVE | {self.dashboard_url}")

    def _stop_monitoring(self) -> None:
        self.stop_event.set()
        self.start_btn.config(text="Start Monitoring")
        self._append_log("WATCHER STOP REQUESTED")

    def _watch_loop(self) -> None:
        writer = _QueueWriter(self.log_queue)
        try:
            with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                while not self.stop_event.is_set():
                    self._run_single_scan_cycle()
                    poll_seconds = int(self.config.get("poll_seconds", 120) or 120)
                    if self.stop_event.wait(timeout=max(30, poll_seconds)):
                        break
        except Exception as exc:  # noqa: BLE001
            self.log_queue.put(f"WATCHER FAILED | {exc}")
        finally:
            self.root.after(0, lambda: self.start_btn.config(text="Start Monitoring"))

    def _run_single_scan_cycle(self) -> None:
        root = Path(self.config.get("database_root") or "")
        if not root.exists():
            self.log_queue.put(f"SCAN SKIPPED | database_root_missing={root}")
            return

        system_root = root / SYSTEM_FOLDER_NAME
        debug_root = system_root / "debug"
        admin_root = system_root / "admin"
        logs_root = system_root / "logs"
        logs_root.mkdir(parents=True, exist_ok=True)
        cycle_rows: List[Dict[str, Any]] = []

        clients = [child for child in sorted(root.iterdir()) if child.is_dir() and not _is_system_folder(child)]
        self.log_queue.put(f"SCAN START | database_root={root} | clients={len(clients)}")

        scan_started = time.perf_counter()
        total_seen = 0
        for client_dir in clients:
            folder_specs = (
                ("Purchase", "Purchase Invoice"),
                ("Sales", "Sales Invoice"),
            )
            for bucket_name, invoice_type in folder_specs:
                bucket_dir = _find_caseflex_subdir(client_dir, bucket_name, bucket_name)
                completed_dir = _find_caseflex_subdir(bucket_dir, "completed", "Completed")
                json_dir = _find_caseflex_subdir(bucket_dir, "json", "JSON")

                input_files = [path for path in sorted(bucket_dir.iterdir()) if _is_supported_input(path) and _is_file_ready(path)]
                if not input_files:
                    continue
                self.log_queue.put(
                    f"CLIENT QUEUE | client={client_dir.name} | bucket={bucket_name} | invoice_type={invoice_type} | files={len(input_files)}"
                )
                batch_app = _load_batch_app(self.log_queue.put)
                for idx, file_path in enumerate(input_files, start=1):
                    total_seen += 1
                    output_json_path = _resolve_unique_path(json_dir / f"{file_path.stem}.json")
                    debug_json_path = _resolve_unique_path(
                        debug_root / client_dir.name / bucket_name / f"{file_path.stem}_debug.json"
                    )
                    result = batch_app.process_invoice_file(
                        file_path=file_path,
                        invoice_type=invoice_type,
                        out_json_path=output_json_path,
                        debug_json_path=debug_json_path,
                        index=idx,
                        total=len(input_files),
                    )
                    result["client_name"] = client_dir.name
                    result["bucket"] = bucket_name
                    result["invoice_type"] = invoice_type
                    result["completed_path"] = None
                    if result.get("status") == "success":
                        moved_to = _resolve_unique_path(completed_dir / file_path.name)
                        shutil.move(str(file_path), str(moved_to))
                        result["completed_path"] = str(moved_to)
                        self.log_queue.put(
                            f"MOVED TO COMPLETED | client={client_dir.name} | bucket={bucket_name} | file={moved_to}"
                        )
                    cycle_rows.append(result)

        duration_ms = int((time.perf_counter() - scan_started) * 1000)
        self._write_admin_cycle(admin_root, cycle_rows, duration_ms)
        self.log_queue.put(
            f"SCAN DONE | processed={len(cycle_rows)} | discovered={total_seen} | duration_ms={duration_ms}"
        )

    def _write_admin_cycle(self, admin_root: Path, rows: List[Dict[str, Any]], duration_ms: int) -> None:
        admin_root.mkdir(parents=True, exist_ok=True)
        history_csv = admin_root / "history.csv"
        latest_scan_json = admin_root / "latest_scan.json"
        latest_dashboard_html = admin_root / "latest_dashboard.html"
        summary = {
            "processed_at": datetime.now().isoformat(timespec="seconds"),
            "duration_ms": duration_ms,
            "files_processed": len(rows),
            "success": sum(1 for row in rows if row.get("status") == "success"),
            "failed": sum(1 for row in rows if row.get("status") != "success"),
            "rows": rows,
        }
        latest_scan_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        latest_dashboard_html.write_text(_render_admin_html(summary, rows), encoding="utf-8")

        headers = [
            "processed_at",
            "client_name",
            "bucket",
            "invoice_type",
            "status",
            "file",
            "output_json",
            "completed_path",
            "debug_json",
            "llm_status",
            "duration_ms",
            "ocr_ms",
            "preprocess_ms",
            "llm_ms",
            "save_ms",
            "invoice_number",
            "invoice_date",
            "supplier_name",
            "customer_name",
            "total_amount",
            "error",
        ]
        write_header = not history_csv.exists()
        with history_csv.open("a", encoding="utf-8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=headers)
            if write_header:
                writer.writeheader()
            stamp = summary["processed_at"]
            for row in rows:
                admin_fields = row.get("admin_fields", {}) if isinstance(row.get("admin_fields"), dict) else {}
                writer.writerow(
                    {
                        "processed_at": stamp,
                        "client_name": row.get("client_name"),
                        "bucket": row.get("bucket"),
                        "invoice_type": row.get("invoice_type"),
                        "status": row.get("status"),
                        "file": row.get("file"),
                        "output_json": row.get("output_json"),
                        "completed_path": row.get("completed_path"),
                        "debug_json": row.get("debug_json"),
                        "llm_status": row.get("llm_status"),
                        "duration_ms": row.get("duration_ms"),
                        "ocr_ms": row.get("ocr_ms"),
                        "preprocess_ms": row.get("preprocess_ms"),
                        "llm_ms": row.get("llm_ms"),
                        "save_ms": row.get("save_ms"),
                        "invoice_number": admin_fields.get("invoice_number"),
                        "invoice_date": admin_fields.get("invoice_date"),
                        "supplier_name": admin_fields.get("supplier_name"),
                        "customer_name": admin_fields.get("customer_name"),
                        "total_amount": admin_fields.get("total_amount"),
                        "error": row.get("error"),
                    }
                )

    def _append_log(self, line: str) -> None:
        self.log_text.insert(tk.END, line + "\n")
        self.log_text.see(tk.END)

    def _poll_logs(self) -> None:
        try:
            while True:
                line = self.log_queue.get_nowait()
                self._append_log(line)
        except queue.Empty:
            pass
        self.root.after(200, self._poll_logs)

    def _create_tray_icon(self) -> None:
        if pystray is None:
            return
        image = Image.new("RGBA", (64, 64), (20, 98, 184, 255))
        draw = ImageDraw.Draw(image)
        draw.rectangle((10, 10, 54, 54), fill=(255, 255, 255, 255))
        draw.rectangle((18, 18, 46, 46), fill=(20, 98, 184, 255))
        menu = pystray.Menu(
            pystray.MenuItem("Open", lambda icon, item: self.root.after(0, self.show_window)),
            pystray.MenuItem("Open Dashboard", lambda icon, item: self.root.after(0, self.open_dashboard)),
            pystray.MenuItem(
                "Start/Stop Monitoring",
                lambda icon, item: self.root.after(0, self._toggle_monitoring),
            ),
            pystray.MenuItem("Exit", lambda icon, item: self.root.after(0, self.exit_app)),
        )
        self.tray_icon = pystray.Icon(APP_NAME, image, "Invoice Batch Watcher", menu)
        threading.Thread(target=self.tray_icon.run, daemon=True).start()

    def hide_to_tray(self) -> None:
        if pystray is None:
            self.root.iconify()
            return
        self.root.withdraw()
        self.log_queue.put("APP HIDDEN TO SYSTEM TRAY")

    def show_window(self) -> None:
        self.root.deiconify()
        self.root.lift()
        self.root.focus_force()

    def _handle_window_close(self) -> None:
        self.hide_to_tray()

    def exit_app(self) -> None:
        self.stop_event.set()
        if self.dashboard_server is not None:
            try:
                self.dashboard_server.shutdown()
                self.dashboard_server.server_close()
            except Exception:
                pass
        if self.tray_icon is not None:
            try:
                self.tray_icon.stop()
            except Exception:
                pass
        self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoice database watcher desktop app")
    parser.add_argument("--start-minimized", action="store_true", help="Start minimized to tray")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    root.withdraw()
    splash = StartupSplash(root)
    splash.set_status("Loading saved configuration...")
    root.update()
    splash.set_status("Starting local dashboard and tray integration...")
    app = WatcherDesktopApp(root, start_minimized=args.start_minimized)
    splash.set_status("Ready.")
    splash.close()
    if not args.start_minimized or pystray is None:
        root.deiconify()
        if args.start_minimized and pystray is None:
            root.iconify()
    root.mainloop()


if __name__ == "__main__":
    main()
