import argparse, random, csv, os, json
from pathlib import Path

VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.webm'}

def gather_files(paths):
    files = []
    for p in paths:
        p = Path(p)
        if not p.exists(): 
            continue
        if p.is_dir():
            for f in p.rglob('*'):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS:
                    files.append(str(f.resolve()))
        elif p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files.append(str(p.resolve()))
    return sorted(files)

def write_csv(rows, outpath):
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    with open(outpath, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['video_path','source','label','split','notes'])
        for r in rows:
            w.writerow(r)

def unique_preserve(seq):
    seen = set(); out=[]
    for s in seq:
        if s not in seen:
            seen.add(s); out.append(s)
    return out

def main(args):
    root = Path(args.root)
    random.seed(args.seed)

    # --- Expected folder layout (matching your screenshot)
    # Celeb-DF
    celeb_v1_real = gather_files([root/'Celeb-DF'/'Celeb-real', root/'Celeb-DF'/'YouTube-real'])
    celeb_v1_synth = gather_files([root/'Celeb-DF'/'Celeb-synthesis'])
    # Celeb-DF-v2
    celeb_v2_real = gather_files([root/'Celeb-DF-v2'/'Celeb-real', root/'Celeb-DF-v2'/'YouTube-real'])
    celeb_v2_synth = gather_files([root/'Celeb-DF-v2'/'Celeb-synthesis'])
    # DFDC
    dfdc = gather_files([
        root/'DFDC'/'dfdc_train_part_00', root/'DFDC'/'dfdc_train_part_01', root/'DFDC'/'dfdc_train_part_02',
        root/'DFDC'/'dfdc_train_part_0', root/'DFDC'/'dfdc_train_part_1', root/'DFDC'/'dfdc_train_part_2',
        root/'DFDC'
    ])
    # SHAM
    sham = gather_files([
        root/'SHAM'/'Processed-Part1', root/'SHAM'/'Processed-Part2',
        root/'SHAM'/'Unedited-Part1', root/'SHAM'/'Unedited-Part2', root/'SHAM'
    ])
    # UCF101 (treat all as real)
    ucf101 = gather_files([root/'UCF101'])
    # VidProM variants
    vid_ms = gather_files([root/'VidProM'/'ms_videos'/'ms_videos_all', root/'VidProM'/'ms_videos', root/'VidProM'/'ms_videos_1'])
    vid_pika = gather_files([root/'VidProM'/'pika_videos'/'pika_videos_all', root/'VidProM'/'pika_videos', root/'VidProM'/'pika_videos_1'])
    vid_t2v = gather_files([root/'VidProM'/'t2vz_videos'/'t2vz_videos_all', root/'VidProM'/'t2v_videos'/'t2v_videos_all', root/'VidProM'/'t2v_videos', root/'VidProM'/'t2v_videos_1', root/'VidProM'/'t2vz_videos'])

    # Flatten + unique
    celeb_v1_real = unique_preserve(celeb_v1_real)
    celeb_v1_synth = unique_preserve(celeb_v1_synth)
    celeb_v2_real = unique_preserve(celeb_v2_real)
    celeb_v2_synth = unique_preserve(celeb_v2_synth)
    dfdc = unique_preserve(dfdc)
    sham = unique_preserve(sham)
    ucf101 = unique_preserve(ucf101)
    vid_ms = unique_preserve(vid_ms)
    vid_pika = unique_preserve(vid_pika)
    vid_t2v = unique_preserve(vid_t2v)

    print("Counts:")
    for k,v in [('celeb_v1_real',celeb_v1_real),('celeb_v1_synth',celeb_v1_synth),
                ('celeb_v2_real',celeb_v2_real),('celeb_v2_synth',celeb_v2_synth),
                ('dfdc',dfdc),('sham',sham),('ucf101',ucf101),
                ('vid_ms',vid_ms),('vid_pika',vid_pika),('vid_t2v',vid_t2v)]:
        print(f"  {k}: {len(v)}")

    # Pools for sampling
    face_synth_pool = celeb_v1_synth + celeb_v2_synth + dfdc + sham
    face_real_pool = celeb_v1_real + celeb_v2_real
    lowq_pool = vid_t2v + vid_ms
    pika_pool = vid_pika
    real_pool = ucf101

    # Parameters
    base_total = args.base_total
    base_val = args.base_val
    base_test = args.base_test
    cl_total = args.cl_total
    pika_holdout = args.pika_holdout

    # Build base (aim to emphasize face manipulations but keep some lowq)
    face_frac = 0.85  # changeable
    n_face = int(base_total * face_frac)
    n_lowq = base_total - n_face

    # Sample face items (mix synth + real to ensure both present)
    face_candidates = face_synth_pool + face_real_pool
    random.shuffle(face_candidates)
    base_face = face_candidates[:n_face]
    random.shuffle(lowq_pool)
    base_lowq = lowq_pool[:n_lowq]

    base_all = base_face + base_lowq
    random.shuffle(base_all)

    # Create rows
    def label_from_path(p):
        # try to infer label by known pools
        if p in face_synth_pool or p in pika_pool or p in lowq_pool:
            return 'fake'
        if p in face_real_pool or p in real_pool:
            return 'real'
        # default
        return 'unknown'

    rows = [(p, 'base_pool', label_from_path(p), '') for p in base_all]
    # split into train / val / test
    total = len(rows)
    n_test = base_test
    n_val = base_val
    n_train = total - n_val - n_test
    train_rows = rows[:n_train]
    val_rows = rows[n_train:n_train+n_val]
    test_rows = rows[n_train+n_val:n_train+n_val+n_test]

    # --- CL arrival composition (revised)
    # composition: 50% pika (HQ), 20% ms (med), 15% real, 15% mixed
    n_pika = int(cl_total * 0.70)
    n_ms = int(cl_total * 0.20)
    n_real = int(cl_total * 0.10)
    n_mixed = 0

    cl_pika = pika_pool[:n_pika]
    cl_ms = vid_ms[:n_ms]
    cl_real = real_pool[:n_real]

    # Create mixed entries by pairing synth + real (note: we store notes; actual mixing happens later)
    mixed_rows = []
    synth_candidates = (vid_ms + face_synth_pool)[:n_mixed]
    for i, p in enumerate(synth_candidates):
        real_choice = real_pool[i % len(real_pool)] if real_pool else ''
        mixed_rows.append((p, 'mixed_pool', 'mixed', f"mixed_with:{Path(real_choice).name if real_choice else ''}"))

    cl_rows = []
    cl_rows += [(p, 'pika', 'fake', 'cl_new_pika') for p in cl_pika]
    cl_rows += [(p, 'vid_ms', 'fake', 'cl_new_ms') for p in cl_ms]
    cl_rows += [(p, 'ucf101', 'real', 'cl_new_real') for p in cl_real]
    cl_rows += mixed_rows

    # Pika holdout
    pika_holdout_list = pika_pool[n_pika:n_pika+pika_holdout] if len(pika_pool) > n_pika else pika_pool[:pika_holdout]
    pika_holdout_rows = [(p, 'pika_holdout', 'fake', 'pika_holdout') for p in pika_holdout_list]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    write_csv(train_rows, outdir/'train_manifest.csv')
    write_csv(val_rows, outdir/'val_manifest.csv')
    write_csv(test_rows, outdir/'test_manifest.csv')
    write_csv(cl_rows, outdir/'cl_arrival_manifest.csv')
    write_csv(pika_holdout_rows, outdir/'pika_holdout_manifest.csv')

    summary = {
        'train': len(train_rows),
        'val': len(val_rows),
        'test': len(test_rows),
        'cl_arrival': len(cl_rows),
        'pika_holdout': len(pika_holdout_rows),
        'params': {
            'base_total': base_total, 'base_val': base_val, 'base_test': base_test, 'cl_total': cl_total, 'pika_holdout': pika_holdout
        }
    }
    with open(outdir/'summary.json','w') as f:
        json.dump(summary, f, indent=2)

    print("Wrote manifests to", outdir)
    print("Summary:", summary)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=str, default='./data', help='root dataset dir')
    p.add_argument('--out', type=str, default='manifests', help='output dir for manifests')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--base_total', type=int, default=2000)
    p.add_argument('--base_val', type=int, default=300)
    p.add_argument('--base_test', type=int, default=500)
    p.add_argument('--cl_total', type=int, default=1000)
    p.add_argument('--pika_holdout', type=int, default=500)
    args = p.parse_args()
    main(args)
