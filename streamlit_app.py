"""Works as `python run_app.py` or a Streamlit Cloud entry point.

Under Streamlit, execute the app in the existing server. Never spawn a second
server from a Streamlit script. Plain Python replaces itself with Streamlit.
"""
from pathlib import Path
import os
import runpy
import sys


def main():
    app_path = Path(__file__).resolve().with_name("app.py")
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx(suppress_warning=True) is not None:
        runpy.run_path(str(app_path), run_name="__main__")
    else:
        os.execv(sys.executable, [sys.executable, "-m", "streamlit", "run",
                                 str(app_path), "--server.headless=true", *sys.argv[1:]])


if __name__ == "__main__":
    main()
