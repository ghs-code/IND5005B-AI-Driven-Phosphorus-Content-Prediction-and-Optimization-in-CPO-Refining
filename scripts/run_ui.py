from pathlib import Path
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

    raise SystemExit(subprocess.call([sys.executable, "-m", "streamlit", "run", str(app_path)]))


if __name__ == "__main__":
    main()
