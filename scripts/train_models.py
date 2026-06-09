import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models.train import train_and_compare


def main() -> None:
    metrics = train_and_compare()
    print("Training complete.")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
