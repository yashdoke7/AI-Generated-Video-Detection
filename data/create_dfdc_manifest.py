import json
import csv
from pathlib import Path
import argparse

p = argparse.ArgumentParser()
p.add_argument('--dfdc_root', required=True)
p.add_argument('--metadata', required=True)
p.add_argument('--output_manifest', required=True)
args = p.parse_args()

dfdc_dir = Path(args.dfdc_root)
metadata_file = Path(args.metadata)
manifest_csv = args.output_manifest

with open(metadata_file, "r") as f:
    metadata = json.load(f)

with open(manifest_csv, "w", newline="", encoding="utf-8") as fout:
    writer = csv.DictWriter(fout, fieldnames=["video_path", "label"])
    writer.writeheader()

    for video_filename, data in metadata.items():
        video_path = dfdc_dir / video_filename
        label = data.get("label", "fake").lower()  # default to 'fake' if missing
        writer.writerow({"video_path": str(video_path), "label": label})

print(f"Manifest created at {manifest_csv}")