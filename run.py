"""Train/validate, explicit held-out evaluation, profiling, report tables."""
import argparse
import csv
import json
import math
import os
import platform
import random
import subprocess
import time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from data import Signs, sha256
from migration_config import validate_relocation
from models import ResNet20, VARIANTS, profile

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    s = torch.initial_seed() % 2**32
    np.random.seed(s); random.seed(s)

def loader(a, split, augment=False, condition='clean', generator=None):
    ds = Signs(a.root, Path(a.splits)/f'{split}.csv', a.size, augment, condition)
    return DataLoader(ds, batch_size=a.batch_size, shuffle=augment, num_workers=a.workers,
                      pin_memory=a.device.startswith('cuda'), worker_init_fn=seed_worker,
                      generator=generator)

def scores(cm):
    tp = cm.diag().double(); cm = cm.double()
    precision=tp/cm.sum(0).clamp(min=1); recall=tp/cm.sum(1).clamp(min=1)
    f1=2*precision*recall/(precision+recall).clamp(min=1e-12)
    return {'accuracy': float(tp.sum()/cm.sum().clamp(min=1)), 'macro_f1':float(f1.mean()),
            'macro_recall':float(recall.mean()), 'samples':int(cm.sum()),
            'per_class':[{'label':i,'support':int(cm[i].sum()),'precision':float(precision[i]),
                          'recall':float(recall[i]),'f1':float(f1[i])} for i in range(len(tp))]}

@torch.inference_mode()
def evaluate(model, dl, device, export=None):
    model.eval(); cm=torch.zeros(43,43,dtype=torch.long); loss_sum=0.; count=0
    rows=[]
    for x,y,paths in dl:
        logits=model(x.to(device)); loss_sum+=float(nn.functional.cross_entropy(logits,y.to(device),reduction='sum'))
        probs=logits.softmax(1).cpu(); pred=probs.argmax(1)
        cm+=torch.bincount(y*43+pred,minlength=43**2).reshape(43,43); count+=len(y)
        if export is not None:
            for p,t,k,prob in zip(paths,y.tolist(),pred.tolist(),probs.max(1).values.tolist()):
                rows.append({'path':p,'true':t,'predicted':k,'max_softmax':prob})
    result=scores(cm); result['loss']=loss_sum/max(1,count)
    if export is not None:
        export=Path(export); export.mkdir(parents=True,exist_ok=True)
        with open(export/'predictions.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=['path','true','predicted','max_softmax']); w.writeheader(); w.writerows(rows)
        np.savetxt(export/'confusion_matrix.csv',cm.numpy(),fmt='%d',delimiter=',')
        save_json(export/'metrics.json', result)
    return result

def training(a):
    out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(a).items() if k not in ('resume','stop_after','allow_relocation')}
    confpath=out/'config.json'
    relocation={}
    previous=None
    if confpath.exists():
        if not a.resume: raise FileExistsError('Run exists. Use --resume or a new --out.')
        previous=json.loads(confpath.read_text())
        relocation=validate_relocation(previous,config,a.allow_relocation)
    seed_all(a.seed)
    gen=torch.Generator().manual_seed(a.seed)
    train=loader(a,'train',True,generator=gen); val=loader(a,'val')
    hashes={s:sha256(Path(a.splits)/f'{s}.csv') for s in ['train','val']}
    model=ResNet20(a.variant)
    stats=profile(model,a.size); model.to(a.device)
    opt=torch.optim.SGD(model.parameters(),lr=a.lr,momentum=.9,weight_decay=5e-4)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=a.epochs)
    history=[]; best=-math.inf; start=0
    if a.resume:
        c=torch.load(out/'last.pt',map_location='cpu',weights_only=False)
        if c['manifest_hashes']!=hashes: raise ValueError('Split manifests changed')
        if relocation:
            record={'old_config':previous,'new_config':config,'changed_paths':relocation,
                    'manifest_hashes':hashes,'torch':torch.__version__,'python':platform.python_version(),
                    'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
                    'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}}
            save_json(out/f'migration_{time.time_ns()}.json',record)
            save_json(confpath,config)
        model.load_state_dict(c['model']); opt.load_state_dict(c['optimizer']); sched.load_state_dict(c['scheduler'])
        history,best,start=c['history'],c['best'],c['epoch']+1
        random.setstate(c['python_rng']); np.random.set_state(c['numpy_rng']); torch.set_rng_state(c['torch_rng'])
        gen.set_state(c['loader_rng'])
        if a.device.startswith('cuda') and c['cuda_rng'] is not None: torch.cuda.set_rng_state_all(c['cuda_rng'])
    else:
        save_json(confpath,config); save_json(out/'profile.json',stats)
        save_json(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,
                  'numpy':np.__version__,'device':a.device,
                  'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
                  'source_sha256':{p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                  'train_val_hashes':hashes})
        (out/'pip_freeze.txt').write_text(subprocess.check_output(['python','-m','pip','freeze'],text=True))
    for epoch in range(start,a.epochs):
        t=time.perf_counter(); model.train(); total=correct=0; loss_sum=0.
        lr=opt.param_groups[0]['lr']
        for x,y,_ in train:
            x,y=x.to(a.device),y.to(a.device)
            opt.zero_grad(set_to_none=True); logits=model(x); loss=nn.functional.cross_entropy(logits,y)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite loss')
            loss.backward(); opt.step()
            loss_sum+=float(loss.detach())*len(y); correct+=int((logits.argmax(1)==y).sum()); total+=len(y)
        v=evaluate(model,val,a.device); sched.step()
        row={'epoch':epoch+1,'lr':lr,'train_loss':loss_sum/total,'train_accuracy':correct/total,
             'val_loss':v['loss'],'val_accuracy':v['accuracy'],'val_macro_f1':v['macro_f1'],
             'epoch_seconds':time.perf_counter()-t}
        history.append(row)
        if v['accuracy']>best:
            best=v['accuracy']
            torch.save({'model':model.state_dict(),'config':config,'epoch':epoch+1,
                        'val_metrics':v,'manifest_hashes':hashes},out/'best.pt')
        checkpoint={'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
                    'epoch':epoch,'best':best,'history':history,'manifest_hashes':hashes,
                    'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),
                    'torch_rng':torch.get_rng_state(),'loader_rng':gen.get_state(),
                    'cuda_rng':torch.cuda.get_rng_state_all() if a.device.startswith('cuda') else None}
        torch.save(checkpoint,out/'last.tmp'); os.replace(out/'last.tmp',out/'last.pt')
        with open(out/'history.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(row)); w.writeheader(); w.writerows(history)
        print(json.dumps(row),flush=True)
        if a.stop_after and epoch+1 >= a.stop_after: break
    save_json(out/'validation_summary.json',{'best_val_accuracy':best,'completed_epochs':len(history),
              'note':'No test data used for training, checkpoint selection, or hyperparameter tuning.'})

def testing(a):
    out=Path(a.out); config=json.loads((out/'config.json').read_text())
    if a.size!=config['size']: raise ValueError('Input size differs from trained checkpoint')
    seed_all(a.seed)
    c=torch.load(out/'best.pt',map_location='cpu',weights_only=False)
    if c['manifest_hashes']['val']!=sha256(Path(a.splits)/'val.csv'): raise ValueError('Wrong validation split')
    model=ResNet20(config['variant']); model.load_state_dict(c['model']); model.to(a.device)
    for condition in ['clean','dark','blur'] if a.robust else ['clean']:
        dest=out/'evaluation'/a.split/condition
        if dest.exists(): raise FileExistsError(f'Evaluation exists: {dest}')
        result=evaluate(model,loader(a,a.split,condition=condition),a.device,dest)
        result.update({'split':a.split,'condition':condition,'checkpoint_sha256':sha256(out/'best.pt'),
                       'manifest_sha256':sha256(Path(a.splits)/f'{a.split}.csv'),
                       'best_epoch':c['epoch'],'synthetic_condition':condition!='clean'})
        save_json(dest/'metrics.json',result); print(condition,result['accuracy'],result['macro_f1'])

def summarize(a):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=[]
    for run in sorted(Path(a.out).glob('*/config.json')):
        run=run.parent; conf=json.loads((run/'config.json').read_text())
        if not (run/'history.csv').exists(): continue
        with open(run/'history.csv') as f: hist=list(csv.DictReader(f))
        fig,axes=plt.subplots(1,2,figsize=(10,4))
        for prefix in ['train','val']:
            axes[0].plot([int(r['epoch']) for r in hist],[float(r[prefix+'_loss']) for r in hist],label=prefix)
            axes[1].plot([int(r['epoch']) for r in hist],[float(r[prefix+'_accuracy']) for r in hist],label=prefix)
        axes[0].set_title('Cross entropy'); axes[1].set_title('Accuracy')
        for ax in axes: ax.set_xlabel('Epoch'); ax.legend()
        fig.tight_layout(); fig.savefig(run/'curves.png',dpi=160); plt.close(fig)
        r={'run':run.name,'variant':conf['variant'],'seed':conf['seed'],
           'best_val_accuracy':max(float(h['val_accuracy']) for h in hist)}
        r.update(json.loads((run/'profile.json').read_text()))
        metric=run/'evaluation/test/clean/metrics.json'
        if metric.exists():
            met=json.loads(metric.read_text()); r.update(test_accuracy=met['accuracy'],test_macro_f1=met['macro_f1'])
        rows.append(r)
    save_json(Path(a.out)/'results.json',rows)
    if rows:
        keys=sorted(set().union(*(r.keys() for r in rows)))
        with open(Path(a.out)/'results.csv','w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows,indent=2))

def main():
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['train','evaluate','profile','summarize'])
    p.add_argument('--root',default='data'); p.add_argument('--splits',default='splits'); p.add_argument('--out',default='runs/baseline_s42')
    p.add_argument('--variant',choices=VARIANTS,default='baseline'); p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=80); p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--lr',type=float,default=.05); p.add_argument('--size',type=int,default=32)
    p.add_argument('--allow-relocation',action='store_true'); p.add_argument('--stop-after',type=int,default=0); p.add_argument('--workers',type=int,default=2); p.add_argument('--resume',action='store_true')
    p.add_argument('--split',choices=['val','test'],default='val'); p.add_argument('--robust',action='store_true')
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    a=p.parse_args()
    if a.action=='train': training(a)
    elif a.action=='evaluate': testing(a)
    elif a.action=='profile':
        rows={v:profile(ResNet20(v),a.size) for v in VARIANTS}; print(json.dumps(rows,indent=2))
    else: summarize(a)
if __name__=='__main__': main()
