from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cpo_phosphorus.pipelines.multi_year_dispatcher import main


if __name__ == "__main__":
    main()
