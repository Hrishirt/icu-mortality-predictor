"""Build a sample API payload from a PhysioNet patient record."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import settings
from src.data.preprocessing import extract_patient_features, load_patient_record


def main() -> None:
    record_id = sys.argv[1] if len(sys.argv) > 1 else "132551"
    record_path = settings.data_dir / "set-a" / f"{record_id}.txt"
    features = extract_patient_features(load_patient_record(record_path))
    payload = {"patient_id": record_id, "features": features}
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
