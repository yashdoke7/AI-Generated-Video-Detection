import argparse
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from torch.utils.data._utils.collate import default_collate

def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

class EmbeddingVideoDataset(Dataset):
    def __init__(self, manifest_csv, emb_root, max_len=64):
        self.rows=[]
        with open(manifest_csv,newline='',encoding='utf-8') as f:
            for r in csv.DictReader(f):
                self.rows.append(r)
        self.emb_root=Path(emb_root); self.max_len=max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self,idx):
        row=self.rows[idx]
        vp=row.get('video_path') or row.get('path') or row.get('filename')
        vid_id=Path(vp).stem
        pths=list(self.emb_root.rglob(f'*{vid_id}*.npy'))
        if not pths: raise FileNotFoundError(vid_id)
        emb=np.load(pths[0])
        if emb.shape[0]>=self.max_len:
            st=(emb.shape[0]-self.max_len)//2
            emb=emb[st:st+self.max_len]
        else:
            pad=np.zeros((self.max_len-emb.shape[0],emb.shape[1]),dtype=emb.dtype)
            emb=np.concatenate([emb,pad],axis=0)
        lbl=row.get('label','fake').lower()
        lab=1 if lbl in ('fake','1') else 0
        return torch.from_numpy(emb.astype('float32')),torch.tensor(lab,dtype=torch.long),vid_id

class TemporalTransformer(nn.Module):
    def __init__(self, emb_dim=768, d_model=128, nhead=4, nlayers=2, ff_dim=256):
        super().__init__()
        self.input_proj=nn.Linear(emb_dim,d_model)
        self.encoder=nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model,nhead,dim_feedforward=ff_dim,batch_first=True
            ),
            nlayers
        )
        self.pool=nn.AdaptiveAvgPool1d(1)
        self.head=nn.Sequential(
            nn.Linear(d_model,64),
            nn.ReLU(),
            nn.Linear(64,1)
        )

    def forward(self,x):
        x=self.input_proj(x)
        x=self.encoder(x)
        y=x.permute(0,2,1)
        y=self.pool(y).squeeze(-1)
        return self.head(y).squeeze(-1)

def evaluate(args):
    device=torch.device(args.device)
    ds=EmbeddingVideoDataset(args.manifest,args.emb_root,args.max_len)
    dl=DataLoader(ds,batch_size=args.batch,shuffle=False,num_workers=args.num_workers,pin_memory=True,collate_fn=collate_skip_none)
    state=torch.load(args.model,map_location='cpu')
    w=state['input_proj.weight']
    emb_dim=w.size(1); d_model=w.size(0)
    model=TemporalTransformer(emb_dim, d_model, args.nhead, args.nlayers, ff_dim=d_model*2)
    model.load_state_dict(state,strict=False)
    model.to(device).eval()

    y_true=[]; y_prob=[]; y_pred=[]; vids=[]
    with torch.no_grad():
        for batch in tqdm(dl):
            if batch is None:
                continue
            x, y, vid = batch
            x=x.to(device)
            logits=model(x)
            probs=torch.sigmoid(logits).cpu().numpy()
            preds=(probs>=args.threshold).astype(int).flatten().tolist()
            y_prob.extend(probs.flatten().tolist())
            y_pred.extend(preds)
            y_true.extend(y.numpy().tolist())
            vids.extend(vid)
    acc=accuracy_score(y_true,y_pred)
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    auc=roc_auc_score(y_true,y_prob) if len(set(y_true))>1 else float('nan')
    cm=confusion_matrix(y_true,y_pred)
    print(f"Samples:{len(y_true)}  Acc:{acc:.4f}  Prec:{p:.4f}  Rec:{r:.4f}  F1:{f1:.4f}  AUC:{auc:.4f}")
    print("CM:\n",cm)
    if args.out_csv:
        import pandas as pd
        pd.DataFrame({'video_id':vids,'label':y_true,'prob':y_prob,'pred':y_pred}).to_csv(args.out_csv,index=False)
        print("Saved preds to",args.out_csv)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',type=str,required=True)
    p.add_argument('--emb_root',type=str,default='embeddings')
    p.add_argument('--model',type=str,default='baseline_transformer_fast.pth')
    p.add_argument('--device',type=str,default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--max_len',type=int,default=64)
    p.add_argument('--nhead',type=int,default=4)
    p.add_argument('--nlayers',type=int,default=2)
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--num_workers',type=int,default=2)
    p.add_argument('--out_csv',type=str,default='')
    args=p.parse_args()
    evaluate(args)
