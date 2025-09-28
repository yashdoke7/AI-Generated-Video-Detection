import argparse, csv, os, subprocess, json, math
from pathlib import Path
import hashlib

def video_id_from_path(p):
    # create short stable id from path
    h = hashlib.sha1(p.encode('utf-8')).hexdigest()[:10]
    name = Path(p).stem
    return f"{name}_{h}"

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def ffprobe_duration(path):
    # returns duration in seconds (float) using ffprobe
    cmd = [
        'ffprobe','-v','error','-select_streams','v:0','-show_entries','format=duration',
        '-of','default=noprint_wrappers=1:nokey=1', path
    ]
    try:
        out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
        return float(out) if out else None
    except Exception:
        return None

def extract_frames_for_video(video_path, out_dir, fps=4, size=224, force=False):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    # if frames already present and not force, skip
    existing = list(out_dir.glob('frame_*.jpg'))
    if existing and not force:
        print(f"Skipping (exists): {video_path} -> {out_dir} ({len(existing)} frames)")
        # still create timeline.json if missing
        return len(existing)

    # ffmpeg command
    vf = f"fps={fps},scale={size}:{size}:flags=lanczos"
    cmd = [
        'ffmpeg','-y','-hide_banner','-loglevel','error',
        '-i', video_path,
        '-vf', vf,
        str(out_dir/'frame_%06d.jpg')
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print("ffmpeg failed for", video_path, e)
        return 0

    frames = sorted(out_dir.glob('frame_*.jpg'))
    return len(frames)

def write_timeline_json(out_dir, fps):
    out_dir = Path(out_dir)
    frames = sorted(out_dir.glob('frame_*.jpg'))
    timeline = []
    for idx, f in enumerate(frames):
        timestamp = idx / fps
        timeline.append({'frame': f.name, 'timestamp': round(timestamp, 6)})
    with open(out_dir/'timeline.json','w', encoding='utf-8') as fh:
        json.dump(timeline, fh, indent=2)
    return len(frames)

def main(args):
    manifest = Path(args.manifest)
    out_root = Path(args.out)
    fps = args.fps
    size = args.size
    force = args.force

    if not manifest.exists():
        print("Manifest not found:", manifest); return

    with open(manifest, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    print(f"Processing {len(rows)} entries from {manifest}")
    count = 0
    for r in rows:
        video_path = r.get('video_path') or r.get('path')
        split = r.get('split', 'unknown') or Path(manifest).stem
        if not Path(video_path).exists():
            print("Missing video:", video_path)
            continue
        vid_id = video_id_from_path(video_path)
        out_dir = out_root/split/vid_id
        ensure_dir(out_dir)
        n = extract_frames_for_video(video_path, out_dir, fps=fps, size=size, force=force)
        if n > 0:
            write_timeline_json(out_dir, fps)
            count += 1
    print("Done. processed:", count)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=str, required=True, help='manifest CSV produced by make_manifests_from_structure.py')
    p.add_argument('--out', type=str, default='frames', help='frames output root')
    p.add_argument('--fps', type=int, default=4, help='frames per second to extract')
    p.add_argument('--size', type=int, default=224, help='resize short edge to this (result will be size x size)')
    p.add_argument('--force', action='store_true', help='re-extract existing outputs')
    args = p.parse_args()
    main(args)
