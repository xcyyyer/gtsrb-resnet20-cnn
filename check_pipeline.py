"""Synthetic fixtures ONLY: functional checks, never coursework accuracy."""
import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from data import write_csv
from models import ResNet20,VARIANTS,profile
from run import scores

def main():
    p=argparse.ArgumentParser(); p.add_argument('--out',default='verification'); a=p.parse_args()
    root=Path(a.out).resolve(); root.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(2)
    profiles={}
    for v in VARIANTS:
        model=ResNet20(v); x=torch.randn(4,3,32,32); logits=model(x)
        assert logits.shape==(4,43)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([0,1,21,42])); loss.backward()
        assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        profiles[v]=profile(model)
    # Metric check with every class represented: one correct and one shifted prediction/class.
    cm=torch.eye(43,dtype=torch.long)
    for i in range(43): cm[i,(i+1)%43]+=1
    s=scores(cm); assert s['accuracy']==.5 and abs(s['macro_f1']-.5)<1e-12
    data=root/'synthetic_data'; splits=root/'synthetic_splits'
    for split,n in [('train',2),('val',1),('test',1)]:
        rows=[]
        for label in range(43):
            for i in range(n):
                path=Path(split)/f'{label}_{i}.png'; (data/path).parent.mkdir(parents=True,exist_ok=True)
                arr=np.random.default_rng(label*10+i).integers(0,256,(32,32,3),dtype=np.uint8)
                Image.fromarray(arr).save(data/path)
                rows.append({'path':str(path),'label':label,'group':f'{split}/{label}'})
        write_csv(splits/f'{split}.csv',rows)
    def command(action,out,extra=()):
        cmd=[sys.executable,'run.py',action,'--root',str(data),'--splits',str(splits),
             '--out',str(out),'--workers','0','--epochs','2','--batch-size','32','--device','cpu',*extra]
        subprocess.run(cmd,check=True)
    for v in VARIANTS:
        out=root/f'{v}_functional'
        command('train',out,['--variant',v,'--stop-after','1'])
        command('train',out,['--variant',v,'--resume'])
        command('evaluate',out,['--split','test','--robust'])
    reference=root/'baseline_uninterrupted'; command('train',reference)
    resumed=torch.load(root/'baseline_functional/last.pt',weights_only=False,map_location='cpu')
    full=torch.load(reference/'last.pt',weights_only=False,map_location='cpu')
    assert all(torch.equal(v,full['model'][k]) for k,v in resumed['model'].items()), 'Resume differs'
    result={'status':'PASS','scope':'SYNTHETIC functional verification; not real GTSRB accuracy',
            'checks':['four architectures forward/backward','known confusion-matrix metrics',
                      'all variants two-epoch training','checkpoint resume matches uninterrupted training',
                      'checkpoint reload and clean/dark/blur evaluation with CSV export'],
            'profiles':profiles,'torch':torch.__version__}
    (root/'verification.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
