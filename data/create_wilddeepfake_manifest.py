import os
import csv
from pathlib import Path
import argparse

def find_frame_folders(test_root, label_type):
    out = []
    for sample_num in os.listdir(test_root):
        sample_path = os.path.join(test_root, sample_num, label_type)
        if not os.path.isdir(sample_path):
            continue
        for subfolder in os.listdir(sample_path):
            frame_folder = os.path.join(sample_path, subfolder)
            if os.path.isdir(frame_folder) and any(f.endswith('.png') for f in os.listdir(frame_folder)):
                out.append({
                    'video_id': f"{label_type}_{sample_num}_{subfolder}",
                    'frames_path': frame_folder,
                    'label': label_type
                })
    return out

def create_wilddeepfake_manifest(root, output_manifest):
    root = Path(root)
    manifest_rows = []

    # Fake frames
    fake_test_path = root / "fake_test" / "fake_test"
    manifest_rows += find_frame_folders(str(fake_test_path), "fake")

    # Real frames
    real_test_path = root / "real_test" / "real_test"
    manifest_rows += find_frame_folders(str(real_test_path), "real")

    # Write manifest
    with open(output_manifest, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['video_id', 'frames_path', 'label'])
        writer.writeheader()
        writer.writerows(manifest_rows)

    print(f"Created WildDeepfake manifest with {len(manifest_rows)} entries: {output_manifest}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wilddeepfake_root', type=str, required=True)
    parser.add_argument('--output_manifest', type=str, default='manifests/wilddeepfake_manifest.csv')
    args = parser.parse_args()
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    create_wilddeepfake_manifest(args.wilddeepfake_root, args.output_manifest)
