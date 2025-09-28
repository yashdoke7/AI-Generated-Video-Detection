import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
from PIL import Image
import json
from transformers import AutoImageProcessor, AutoModel

def load_image_paths(video_dir):
    p = Path(video_dir)
    files = sorted([str(f) for f in p.glob("frame_*.jpg")])
    return files

def batchify(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def pil_load(path, size=None):
    img = Image.open(path).convert("RGB")
    if size is not None:
        img = img.resize((size, size), Image.BICUBIC)
    return img

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--frames_root', type=str, default='frames')
    p.add_argument('--out_root', type=str, default='embeddings')
    p.add_argument('--model_name', type=str, default='google/siglip2-base-patch16-224')
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--fp16', action='store_true', help='use float16 on model + save embeddings as float16')
    p.add_argument('--size', type=int, default=224, help='resize frames to this square size before inference')
    p.add_argument('--all_splits', action='store_true', help='process all subfolders under frames_root')
    p.add_argument('--split', type=str, default=None, help='process a single split folder name under frames_root')
    p.add_argument('--skip_existing', action='store_true', help='skip videos with existing .npy')
    args = p.parse_args()

    frames_root = Path(args.frames_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Validate model + processor
    processor = AutoImageProcessor.from_pretrained(args.model_name)
    print(f"Loading model {args.model_name} with trust_remote_code=True (may download large files)...")
    model = AutoModel.from_pretrained(args.model_name, trust_remote_code=True)
    model.to(args.device)
    model.eval()
    if args.fp16 and args.device.startswith('cuda'):
        model.half()


    # Decide splits to process
    if args.all_splits:
        splits = [d.name for d in sorted(frames_root.iterdir()) if d.is_dir()]
    elif args.split:
        splits = [args.split]
    else:
        # default: typical splits present
        splits = ['train_manifest', 'val_manifest', 'test_manifest', 'cl_arrival', 'pika_holdout', 'aegis_manifest', 'dfdc_test_manifest']
        # but filter to existing
        splits = [s for s in splits if (frames_root / s).exists()]

    print("Splits to process:", splits)

    for split in splits:
        split_dir = frames_root / split
        if not split_dir.exists():
            print("Skipping missing split:", split_dir); continue
        print(f"Processing split {split} -> {split_dir}")
        # each child is a video folder
        vids = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        for vid_dir in tqdm(vids, desc=f"{split}", unit='vid'):
            vid_id = vid_dir.name
            out_path = out_root / split / (vid_id + '.npy')
            meta_path = out_root / split / (vid_id + '.json')
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if args.skip_existing and out_path.exists():
                continue

            frames = load_image_paths(vid_dir)
            if not frames:
                # no frames extracted for this video
                continue

            all_embs = []
            # process in small batches (batch default 1 for low VRAM)
            for batch in batchify(frames, args.batch):
                imgs = [pil_load(p, size=args.size) for p in batch]
                inputs = processor(images=imgs, return_tensors='pt')
                pixel_values = inputs['pixel_values'].to(args.device)
                if args.fp16 and args.device.startswith('cuda'):
                    pixel_values = pixel_values.half()
                with torch.no_grad():
                    # For SigLIP/CLIP, use the vision tower explicitly
                    vision_outputs = model.vision_model(pixel_values=pixel_values)
                    feat = vision_outputs.pooler_output.cpu().numpy() \
                    if getattr(vision_outputs, 'pooler_output', None) is not None \
                    else vision_outputs.last_hidden_state.mean(dim=1).cpu().numpy()

                all_embs.append(feat)

            if not all_embs:
                continue
            emb = np.concatenate(all_embs, axis=0)
            # save float16 to save disk
            emb_to_save = emb.astype(np.float16)
            np.save(out_path, emb_to_save)

            # save meta: original frame count, embedding shape, timestamps (read timeline.json if present)
            timeline_file = vid_dir / 'timeline.json'
            timestamps = None
            if timeline_file.exists():
                try:
                    timestamps = json.load(open(timeline_file, 'r'))
                except Exception:
                    timestamps = None
            meta = {
                'video_id': vid_id,
                'n_frames': emb.shape[0],
                'embedding_dim': emb.shape[1],
                'frames_dir': str(vid_dir),
                'timeline': timestamps
            }
            with open(meta_path, 'w', encoding='utf-8') as fh:
                json.dump(meta, fh, indent=2)

    print("Done. Embeddings saved to:", out_root)

if __name__ == '__main__':
    main()