"""Download the math and Portuguese student CSVs from Kaggle."""

import argparse
import shutil
import tempfile
from pathlib import Path

import kagglehub

DATASET = "uciml/student-alcohol-consumption"
FILES = ("student-mat.csv", "student-por.csv")
ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Download expected CSVs into the ignored data/raw directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "raw")
    output_dir = parser.parse_args().output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="student-data-") as temporary:
        download = Path(kagglehub.dataset_download(DATASET, output_dir=temporary))
        for filename in FILES:
            match = next((path for path in download.rglob(filename) if path.is_file()), None)
            if match is None:
                raise FileNotFoundError(f"{filename} not found in the Kaggle download")
            destination = output_dir / filename
            shutil.copyfile(match, destination)
            print(f"Saved {destination}")


if __name__ == "__main__":
    main()
