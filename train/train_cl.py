import argparse, random, csv, copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from tqdm import tqdm

# Skip None samples
def collate_skip_none(batch):
    batch = [b for b in batch if b is not None]
    return default_collate(batch) if batch else None

class EmbeddingVideoDataset(Dataset):
    def __init__(self, manifest_csv, emb_root, max_len=32):
        self.rows = list(csv.DictReader(open(manifest_csv, newline='', encoding='utf-8')))
        self.emb_root = Path(emb_root)
        self.max_len = max_len
        random.seed(0)
    def __len__(self): return len(self.rows)
    def _find(self, vid_id):
        return next(iter(self.emb_root.rglob(f'*{vid_id}*.npy')), None)
    def __getitem__(self, idx):
        row = self.rows[idx]
        vid = Path(row.get('video_path') or row.get('filename')).stem
        p = self._find(vid)
        if not p: return None
        emb = np.load(p)
        if emb.shape[0] >= self.max_len:
            i = 0 if emb.shape[0]==self.max_len else random.randint(0, emb.shape[0]-self.max_len)
            emb = emb[i:i+self.max_len]
        else:
            pad = np.zeros((self.max_len-emb.shape[0], emb.shape[1]), emb.dtype)
            emb = np.concatenate([emb, pad], axis=0)
        label = 1 if row.get('label','fake').lower() in ('fake','1') else 0
        return torch.from_numpy(emb.astype('float32')), torch.tensor(label, dtype=torch.float32)

class FastTemporalTransformer(nn.Module):
    def __init__(self, emb_dim=768, d_model=128, nhead=4, nlayers=2, max_len=32, diversity_weight=0.3):
        super().__init__()
        self.input_proj = nn.Linear(emb_dim, d_model)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=d_model*2, batch_first=True, dropout=0.1),
            nlayers
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(nn.Linear(d_model,64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64,1))
        self.div_w = diversity_weight

    def forward(self, x, return_div=False):
        x = self.input_proj(x)             # B,T,D
        out = self.encoder(x)              # B,T,D
        y = out.permute(0,2,1)             # B,D,T
        y = self.pool(y).squeeze(-1)       # B,D
        logits = self.head(y).squeeze(-1)
        if return_div:
            attn = torch.norm(out,dim=-1)  # B,T
            probs = torch.softmax(attn,dim=-1)
            div = self.div_w * (probs**2).sum(dim=-1).mean()
            return logits, div
        return logits

class FastReservoirBuffer:
    def __init__(self, cap, seed=42):
        self.cap, self.buf, self.n = cap, [], 0
        random.seed(seed)
    def add(self,s):
        self.n+=1
        if len(self.buf)<self.cap: self.buf.append(s)
        else:
            i = random.randint(0,self.n-1)
            if i<self.cap: self.buf[i]=s
    def sample(self,k):
        from random import sample
        return sample(self.buf, min(k,len(self.buf))) if self.buf else []

def train_task(model,opt,crit,dl,dev,buf,rr,teacher,kdl):
    model.train()
    scaler = torch.amp.GradScaler()
    tloss=0.0
    for batch in tqdm(dl):
        if batch is None: continue
        x,y = batch; x,y=x.to(dev),y.to(dev)
        opt.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            logits,div = model(x,return_div=True)
            main = crit(logits,y)
            rep=0.0
            if buf and rr>0:
                rb = buf.sample(max(1,int(x.size(0)*rr)))
                if rb:
                    embs = torch.stack([torch.from_numpy(r[0]) for r in rb],dim=0).to(dev)
                    lbs  = torch.tensor([r[1] for r in rb],dtype=torch.float32,device=dev)
                    rep = crit(model(embs).squeeze(-1),lbs)
            kd=0.0
            if teacher:
                with torch.no_grad():
                    tlog = teacher(x).detach()
                    tprob= torch.sigmoid(tlog).squeeze(-1)
                kd = nn.BCEWithLogitsLoss()(logits,tprob)
            loss = main+rep+kdl*kd+div
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        tloss+=loss.item()
    return tloss/max(1,len(dl))

def main(args):
    dev=args.device
    ds=EmbeddingVideoDataset(args.tasks,args.emb_root,args.max_len)
    dl=DataLoader(ds,batch_size=args.batch,shuffle=True,num_workers=4,pin_memory=True,collate_fn=collate_skip_none)
    model = FastTemporalTransformer(args.emb_dim,args.d_model,args.nhead,args.nlayers,args.max_len,args.replay_ratio).to(dev)
    if args.use_adapters:
        from adapters import attach_adapters
        model=attach_adapters(model,bottleneck=args.adapter_bottleneck)
    if args.init_model:
        sd=torch.load(args.init_model,map_location='cpu')
        try: model.load_state_dict(sd)
        except: model.load_state_dict(sd,strict=False)
    crit=nn.BCEWithLogitsLoss()
    opt=torch.optim.AdamW(model.parameters(),lr=args.lr)
    sched=torch.optim.lr_scheduler.StepLR(opt,step_size=2,gamma=0.8)
    buf=FastReservoirBuffer(args.buffer_size)
    teacher=None
    if args.lwf: teacher=copy.deepcopy(model).eval().to(dev)
    for ep in range(args.epochs_per_task):
        loss = train_task(model,opt,crit,dl,dev,buf,args.replay_ratio,teacher,args.kd_lambda)
        print(f"Epoch {ep+1}/{args.epochs_per_task} loss {loss:.4f}")
        sched.step()
        # refill buffer
        for i in random.sample(range(len(ds)),min(len(ds),args.buffer_add_per_epoch)):
            e,l = ds[i]
            buf.add((e.numpy(),l.item()))
    ckpt=Path(args.out_dir)/f"traincl_fast_task1.pth"
    torch.save(model.state_dict(),ckpt); print("Saved",ckpt)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--tasks',required=True)
    p.add_argument('--emb_root',default='embeddings')
    p.add_argument('--init_model',default='')
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs_per_task',type=int,default=5)
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--buffer_size',type=int,default=300)
    p.add_argument('--buffer_add_per_epoch',type=int,default=30)
    p.add_argument('--replay_ratio',type=float,default=0.3)
    p.add_argument('--lwf',action='store_true')
    p.add_argument('--kd_lambda',type=float,default=0.5)
    p.add_argument('--max_len',type=int,default=32)
    p.add_argument('--emb_dim',type=int,default=768)
    p.add_argument('--d_model',type=int,default=128)
    p.add_argument('--nhead',type=int,default=4)
    p.add_argument('--nlayers',type=int,default=2)
    p.add_argument('--out_dir',default='checkpoints_fast')
    p.add_argument('--use_adapters',action='store_true')
    p.add_argument('--adapter_bottleneck',type=int,default=32)
    args=p.parse_args()
    Path(args.out_dir).mkdir(exist_ok=True)
    main(args)
