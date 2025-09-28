import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm

class EmbeddingVideoDataset(Dataset):
    def __init__(self, manifest_csv, emb_root, max_len=64):
        self.rows = []
        with open(manifest_csv, newline='', encoding='utf-8') as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append(row)
        self.emb_root = Path(emb_root)
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        path = row.get('video_path') or row.get('path') or row.get('filename')
        vid_id = Path(path).stem
        candidate = list(self.emb_root.rglob(f'*{vid_id}*.npy'))
        if not candidate:
            raise FileNotFoundError(f'Embedding not found for {vid_id}')
        emb = np.load(candidate[0])
        if emb.shape[0] >= self.max_len:
            start = np.random.randint(0, emb.shape[0] - self.max_len + 1)
            emb = emb[start:start + self.max_len]
        else:
            pad = np.zeros((self.max_len - emb.shape[0], emb.shape[1]), dtype=emb.dtype)
            emb = np.concatenate([emb, pad], axis=0)
        label = 1 if row.get('label','fake').lower() == 'fake' else 0
        return torch.from_numpy(emb.astype('float32')), torch.tensor(label, dtype=torch.float32)

class FastTemporalTransformer(nn.Module):
    def __init__(self, emb_dim=768, d_model=128, nhead=4, nlayers=2, max_len=64, diversity_weight=0.15):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model, nhead,
                dim_feedforward=d_model*2,
                batch_first=True,
                dropout=0.1
            ),
            nlayers
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
        self.diversity_weight = diversity_weight

    def forward(self, x, return_diversity_loss=False):
        x = self.input_proj(x)           # [B,T,D]
        out = self.encoder(x)            # [B,T,D]
        y = out.permute(0,2,1)           # [B,D,T]
        y = self.pool(y).squeeze(-1)     # [B,D]
        logits = self.head(y).squeeze(-1)
        if return_diversity_loss:
            attn_norm = torch.norm(out, dim=-1)      # [B,T]
            probs = torch.softmax(attn_norm, dim=-1) # [B,T]
            concentration = torch.sum(probs**2, dim=-1)
            div_loss = self.diversity_weight * concentration.mean()
            return logits, div_loss
        return logits

def train(args):
    ds = EmbeddingVideoDataset(args.manifest, args.emb_root, max_len=args.max_len)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4, pin_memory=True)
    model = FastTemporalTransformer(
        emb_dim=args.emb_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        nlayers=args.nlayers,
        max_len=args.max_len,
        diversity_weight=args.attention_diversity
    ).to(args.device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt,
        max_lr=args.lr*3,
        epochs=args.epochs,
        steps_per_epoch=len(dl),
        pct_start=0.3
    )
    loss_fn = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler()

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(dl, desc=f'Epoch {epoch+1}/{args.epochs}')
        tot_loss = tot_main = tot_div = 0.0
        for i, (x,y) in enumerate(pbar):
            x,y = x.to(args.device,non_blocking=True), y.to(args.device,non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast(device_type='cuda'):
                logits, div_loss = model(x, return_diversity_loss=True)
                main_loss = loss_fn(logits, y)
                loss = main_loss + div_loss
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            tot_loss+=loss.item(); tot_main+=main_loss.item(); tot_div+=div_loss.item()
            if (i+1)%50==0:
                pbar.set_postfix({
                    'Loss':f'{tot_loss/(i+1):.4f}',
                    'Main':f'{tot_main/(i+1):.4f}',
                    'Div': f'{tot_div/(i+1):.4f}',
                    'LR': f'{scheduler.get_last_lr()[0]:.2e}'
                })
    if args.out:
        torch.save(model.state_dict(), args.out)
        print('Saved', args.out)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=str,default='manifests/train_manifest.csv')
    p.add_argument('--emb_root',type=str,default='embeddings')
    p.add_argument('--out',type=str,default='baseline_transformer_fast.pth')
    p.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max_len',type=int,default=64)
    p.add_argument('--emb_dim',type=int,default=768)
    p.add_argument('--d_model',type=int,default=128)
    p.add_argument('--nhead',type=int,default=4)
    p.add_argument('--nlayers',type=int,default=2)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=2e-4)
    p.add_argument('--weight_decay',type=float,default=0.01)
    p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--attention_diversity',type=float,default=0.15)
    args=p.parse_args()
    train(args)
