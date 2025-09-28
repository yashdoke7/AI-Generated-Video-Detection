import argparse, torch, numpy as np, pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
from train.train_cl import FastTemporalTransformer, EmbeddingVideoDataset

def evaluate(args):
    dev=torch.device(args.device)
    # infer dims from checkpoint
    ckpt=torch.load(args.model,map_location='cpu')
    w=ckpt['input_proj.weight']
    d_model,emb_dim=w.size(0),w.size(1)
    model=FastTemporalTransformer(emb_dim,d_model,args.nhead,args.nlayers,args.max_len,args.replay_ratio)
    if args.use_adapters:
        from train.adapters import attach_adapters
        model=attach_adapters(model,bottleneck=args.adapter_bottleneck)
    model.load_state_dict(ckpt,strict=False)
    model.to(dev).eval()
    ds=EmbeddingVideoDataset(args.manifest,args.emb_root,args.max_len)
    dl=DataLoader(ds,batch_size=args.batch,shuffle=False,num_workers=args.num_workers)
    y_true,y_prob,y_pred,vids=[],[],[],[]
    with torch.no_grad():
        for x,y in tqdm(dl):
            x,y=x.to(dev),y.to(dev)
            logits=model(x).cpu()
            probs=torch.sigmoid(logits).numpy()
            preds=(probs>=args.threshold).astype(int)
            y_true+=y.cpu().numpy().tolist()
            y_prob+=probs.tolist()
            y_pred+=preds.tolist()
    acc=accuracy_score(y_true,y_pred)
    p,r,f1,_=precision_recall_fscore_support(y_true,y_pred,average='binary',zero_division=0)
    auc=roc_auc_score(y_true,y_prob) if len(set(y_true))>1 else float('nan')
    cm=confusion_matrix(y_true,y_pred)
    print(f"Acc:{acc:.4f} Prec:{p:.4f} Rec:{r:.4f} F1:{f1:.4f} AUC:{auc:.4f}", "\nCM:\n",cm)
    if args.out_csv:
        pd.DataFrame({'label':y_true,'prob':y_prob,'pred':y_pred}).to_csv(args.out_csv,index=False)
        print("Saved",args.out_csv)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--manifest',required=True)
    p.add_argument('--emb_root',required=True)
    p.add_argument('--model',required=True)
    p.add_argument('--out_csv',required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch',type=int,default=8)
    p.add_argument('--max_len',type=int,default=32)
    p.add_argument('--nhead',type=int,default=4)
    p.add_argument('--nlayers',type=int,default=2)
    p.add_argument('--threshold',type=float,default=0.5)
    p.add_argument('--num_workers',type=int,default=2)
    p.add_argument('--use_adapters',action='store_true')
    p.add_argument('--adapter_bottleneck',type=int,default=64)
    p.add_argument('--replay_ratio',type=float,default=0.3)
    args=p.parse_args()
    evaluate(args)
