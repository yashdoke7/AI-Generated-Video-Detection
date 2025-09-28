import csv
from pathlib import Path
import argparse

def create_aegis_manifest_from_videos(aegis_root, output_manifest):
    aegis_path = Path(aegis_root)
    video_root = aegis_path / "hard_test_set_video" / "test_data"
    
    manifest_rows = []
    
    # Process AI-generated videos (fake)
    ai_gen_path = video_root / "ai_gen"
    for generator in ["kling", "sora"]:
        gen_path = ai_gen_path / generator
        if gen_path.exists():
            for video_file in gen_path.glob("*.mp4"):
                manifest_rows.append({
                    'video_path': str(video_file),
                    'label': 'fake'
                })
    
    # Process real videos
    real_path = video_root / "real"
    for source in ["dvf", "youtube"]:
        source_path = real_path / source
        if source_path.exists():
            for video_file in source_path.glob("*.mp4"):
                manifest_rows.append({
                    'video_path': str(video_file),
                    'label': 'real'
                })
    
    # Write manifest
    with open(output_manifest, 'w', newline='', encoding='utf-8') as f:
        if manifest_rows:
            writer = csv.DictWriter(f, fieldnames=['video_path', 'label'])
            writer.writeheader()
            writer.writerows(manifest_rows)
    
    print(f"Created AEGIS manifest with {len(manifest_rows)} videos: {output_manifest}")
    print(f"Fake videos: {sum(1 for r in manifest_rows if r['label'] == 'fake')}")
    print(f"Real videos: {sum(1 for r in manifest_rows if r['label'] == 'real')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--aegis_root', type=str, required=True)
    parser.add_argument('--output_manifest', type=str, default='manifests/aegis_videos_manifest.csv')
    
    args = parser.parse_args()
    Path(args.output_manifest).parent.mkdir(parents=True, exist_ok=True)
    create_aegis_manifest_from_videos(args.aegis_root, args.output_manifest)
