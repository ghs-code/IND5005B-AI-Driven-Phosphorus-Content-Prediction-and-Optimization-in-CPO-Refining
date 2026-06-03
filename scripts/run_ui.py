from pathlib import Path
import os
import subprocess
import sys


def main():
    app_path = Path(__file__).resolve().parents[1] / "src" / "cpo_phosphorus" / "ui" / "workbench.py"
    try:
        import streamlit  # noqa: F401
    except ModuleNotFoundError:
        print(
            "Streamlit is not installed. Install the project with: "
            "python -m pip install -e .",
            file=sys.stderr,
        )
        raise SystemExit(1)

    env = os.environ.copy()
    env.setdefault("STREAMLIT_LOGGER_LEVEL", "error")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--logger.level=error",
    ]
    try:
        completed = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL, check=False)
    except KeyboardInterrupt:
        raise SystemExit(0)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
