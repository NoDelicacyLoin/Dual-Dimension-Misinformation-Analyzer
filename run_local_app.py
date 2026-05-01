import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.join(PROJECT_ROOT, "backend")
FRONTEND_ROOT = os.path.join(PROJECT_ROOT, "frontend")

LOCAL_TAVILY_API_KEY = "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"
LOCAL_GEMINI_API_KEY = "AIzaSyC5CfBmlrTDBPMqoDMe4OUaW_ODSduG_Lk"


def apply_local_api_keys() -> None:
    if LOCAL_TAVILY_API_KEY:
        os.environ["TAVILY_API_KEY"] = LOCAL_TAVILY_API_KEY
    if LOCAL_GEMINI_API_KEY:
        os.environ["GEMINI_API_KEY"] = LOCAL_GEMINI_API_KEY


def check_required_files() -> None:
    required_files = [
        os.path.join(BACKEND_ROOT, "app.py"),
        os.path.join(BACKEND_ROOT, "analysis_orchestrator.py"),
        os.path.join(FRONTEND_ROOT, "index.html"),
        os.path.join(FRONTEND_ROOT, "script.js"),
        os.path.join(FRONTEND_ROOT, "style.css"),
    ]
    missing_files = [file_path for file_path in required_files if not os.path.exists(file_path)]
    if missing_files:
        raise FileNotFoundError("Missing required project files:\n" + "\n".join(missing_files))


def show_runtime_status() -> None:
    print("Project root:", PROJECT_ROOT)
    print("Backend root:", BACKEND_ROOT)
    print("Frontend root:", FRONTEND_ROOT)
    print("TAVILY_API_KEY:", "set" if os.environ.get("TAVILY_API_KEY") else "missing")
    print("GEMINI_API_KEY:", "set" if os.environ.get("GEMINI_API_KEY") else "missing")

    sys.path.insert(0, BACKEND_ROOT)
    from fact_checking.gemini_agent import is_gemini_available

    print("Gemini available:", is_gemini_available())


def main() -> None:
    apply_local_api_keys()
    check_required_files()
    show_runtime_status()

    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("uvicorn is not installed in this Python environment.") from error

    host = "127.0.0.1"
    port = int(os.environ.get("APP_PORT", "8000"))

    os.chdir(BACKEND_ROOT)
    print()
    print(f"Starting app at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    print()

    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
