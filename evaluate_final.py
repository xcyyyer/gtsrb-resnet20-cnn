"""Fixed held-out evaluation of all 12 selected GTSRB checkpoints; never trains."""
import argparse, csv, hashlib, json, os, platform, statistics, time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset, DataLoader
from models import ResNet20

VARIANTS=['baseline','ds','se','ds_se']
SEEDS=[42,10,20]
CONDITIONS=['clean','dark','blur']
HASHES={'train':'e513ec0114426d5b6a744e499b42f169064ff5152055ada3a7f611f14fb641fe',
        'val':'0353104339cdb38107222c379841c734f40511e87c46190580044a819a3838db',
        'test':'3110a6d095cd25c7d7a19d3739dc99d2dc699ecabb1b05d54acc81aed0d046c9'}
PROTOCOL={'split':'test','samples':12630,'classes':43,'size':32,
    'resize':'PIL RGB then BILINEAR to 32x32','normalization':'(x/255 - 0.5)/0.5',
    'conditions':{'clean':'none','dark':'PIL ImageEnhance.Brightness factor 0.5 AFTER resize',
                  'blur':'PIL GaussianBlur radius 1 AFTER resize'},
    'checkpoint_selection':'maximum validation accuracy; first epoch in ties',
    'seeds':SEEDS,'variants':VARIANTS,
    'purpose':'Evaluate all models without training, reselection or test-driven tuning.'}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):
    with open(p,newline='',encoding='utf-8-sig') as f:return list(csv.DictReader(f))
def write_csv(p,rows):
    with open(p,'w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def save_json(p,data):
    p=Path(p);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2));os.replace(tmp,p)
def scores(cm):
    cm=np.asarray(cm,dtype=np.int64); tp=cm.diagonal().astype(float)
    precision=np.divide(tp,cm.sum(0),out=np.zeros(43),where=cm.sum(0)>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(43),where=cm.sum(1)>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(43),where=(precision+recall)>0)
    return {'accuracy':float(tp.sum()/cm.sum()),'macro_f1':float(f1.mean()),
        'macro_recall':float(recall.mean()),'samples':int(cm.sum()),
        'per_class':[{'label':i,'support':int(cm[i].sum()),'precision':float(precision[i]),'recall':float(recall[i]),'f1':float(f1[i])} for i in range(43)]}

class CachedSigns(Dataset):
    def __init__(self,images,labels):self.images,self.labels=images,labels
    def __len__(self):return len(self.labels)
    def __getitem__(self,i):
        arr=np.array(self.images[i],dtype=np.float32).transpose(2,0,1)/255.
        return torch.from_numpy((arr-.5)/.5),int(self.labels[i])

def validate(study,root):
    for s,h in HASHES.items():assert sha(study/'splits'/f'{s}.csv')==h,f'Wrong {s} manifest'
    entries=read_csv(study/'splits/test.csv'); assert len(entries)==12630
    missing=[r['path'] for r in entries if not (root/r['path']).is_file()]
    assert not missing,f'Missing original images, examples: {missing[:3]}'
    ckhashes={}
    for v in VARIANTS:
        for seed in SEEDS:
            run=study/'runs'/f'{v}_s{seed}'
            hist=read_csv(run/'history.csv');assert [int(r['epoch']) for r in hist]==list(range(1,81)),f'Incomplete run: {run.name}'
            cfg=json.loads((run/'config.json').read_text())
            assert (cfg['variant'],cfg['seed'],cfg['size'],cfg['epochs'])==(v,seed,32,80)
            best=max(hist,key=lambda r:float(r['val_accuracy']))
            ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            assert ck['epoch']==int(best['epoch'])
            assert abs(ck['val_metrics']['accuracy']-float(best['val_accuracy']))<1e-12
            assert ck['manifest_hashes']=={s:HASHES[s] for s in ['train','val']}
            assert ck['config']['variant']==v and ck['config']['seed']==seed
            model=ResNet20(v);model.load_state_dict(ck['model'],strict=True)
            ckhashes[run.name]=sha(run/'best.pt')
    return entries,ckhashes

def preload(root,entries):
    images={c:[] for c in CONDITIONS}
    for i,r in enumerate(entries):
        with Image.open(root/r['path']) as src:im=src.convert('RGB').resize((32,32),Image.Resampling.BILINEAR)
        images['clean'].append(np.array(im,dtype=np.uint8))
        images['dark'].append(np.array(ImageEnhance.Brightness(im).enhance(.5),dtype=np.uint8))
        images['blur'].append(np.array(im.filter(ImageFilter.GaussianBlur(1.0)),dtype=np.uint8))
        if (i+1)%3000==0:print(f'Prepared images: {i+1}/{len(entries)}',flush=True)
    return {c:np.stack(a) for c,a in images.items()}

@torch.inference_mode()
def evaluate_one(model,images,entries,device,batch_size,workers):
    labels=np.array([int(r['label']) for r in entries],dtype=np.int64)
    dl=DataLoader(CachedSigns(images,labels),batch_size=batch_size,num_workers=workers,shuffle=False,pin_memory=device.startswith('cuda'))
    preds=[];conf=[];loss=0.;model.eval();start=time.perf_counter()
    for x,y in dl:
        logits=model(x.to(device));loss+=float(torch.nn.functional.cross_entropy(logits,y.to(device),reduction='sum'))
        prob=logits.softmax(1);p=prob.argmax(1)
        preds.extend(p.cpu().tolist());conf.extend(prob.max(1).values.cpu().tolist())
    cm=np.bincount(labels*43+np.array(preds),minlength=43**2).reshape(43,43)
    metrics=scores(cm);metrics['loss']=loss/len(entries)
    metrics['evaluation_seconds']=time.perf_counter()-start
    rows=[{'path':r['path'],'true':int(r['label']),'predicted':int(p),'max_softmax':float(q)} for r,p,q in zip(entries,preds,conf)]
    return metrics,cm,rows

def figures(out,all_rows,images,entries):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2456A6','#168477','#D18A21','#8956A3']; names=['ResNet-20','DS','SE','DS + SE']
    dest=out/'figures';dest.mkdir(exist_ok=True)
    def save(fig,name):
        fig.savefig(dest/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(dest/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
    agg=[]
    for condition in CONDITIONS:
        for v in VARIANTS:
            rows=[r for r in all_rows if r['variant']==v and r['condition']==condition]
            agg.append({'variant':v,'condition':condition,'n_seeds':3,
                'accuracy_mean_pct':statistics.mean(r['accuracy']*100 for r in rows),
                'accuracy_sample_std_pp':statistics.stdev(r['accuracy']*100 for r in rows),
                'macro_f1_mean_pct':statistics.mean(r['macro_f1']*100 for r in rows),
                'macro_f1_sample_std_pp':statistics.stdev(r['macro_f1']*100 for r in rows)})
    write_csv(out/'test_three_seeds.csv',agg)
    fig,axes=plt.subplots(1,2,figsize=(12,4.5))
    for ax,mean,sd,title in [(axes[0],'accuracy_mean_pct','accuracy_sample_std_pp','Test accuracy'),(axes[1],'macro_f1_mean_pct','macro_f1_sample_std_pp','Test Macro-F1')]:
        for i,v in enumerate(VARIANTS):
            rows=[next(r for r in agg if r['variant']==v and r['condition']==c) for c in CONDITIONS]
            x=np.arange(3)+(i-1.5)*.17
            ax.errorbar(x,[r[mean] for r in rows],yerr=[r[sd] for r in rows],fmt='o-',color=colors[i],label=names[i],capsize=4)
        ax.set_xticks(range(3),CONDITIONS);ax.set(title=title,ylabel='Percent (%)');ax.grid(alpha=.2);ax.legend()
    fig.suptitle('Official test set: 3 seeds, mean ± sample SD; dark/blur are synthetic');fig.tight_layout();save(fig,'test_conditions_mean_std')
    # One fixed seed shown for all models, without choosing a seed on test accuracy.
    fig,axes=plt.subplots(2,2,figsize=(15,13))
    for ax,v,title in zip(axes.flat,VARIANTS,names):
        cm=np.loadtxt(out/f'{v}_s42/clean/confusion_matrix.csv',delimiter=',')
        norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        im=ax.imshow(norm,vmin=0,vmax=1,cmap='Blues');ax.set(title=f'{title}, seed 42',xlabel='Predicted label',ylabel='True label')
        ax.set_xticks(range(0,43,5));ax.set_yticks(range(0,43,5));fig.colorbar(im,ax=ax,fraction=.045)
    fig.suptitle('Clean test confusion matrices; rows normalized (same fixed seed)');fig.tight_layout();save(fig,'clean_confusion_seed42')
    fig,ax=plt.subplots(figsize=(15,4.5))
    for i,v in enumerate(VARIANTS):
        per=[]
        for s in SEEDS:
            met=json.loads((out/f'{v}_s{s}/clean/metrics.json').read_text());per.append([r['f1']*100 for r in met['per_class']])
        ax.plot(range(43),np.mean(per,axis=0),marker='.',lw=1,label=names[i],color=colors[i])
    ax.set(xlabel='Class label',ylabel='Macro over seeds of per-class F1 (%)',title='Clean test: class-wise F1 averaged over 3 seeds');ax.set_xticks(range(43));ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,'clean_per_class_f1')
    fig,axes=plt.subplots(3,8,figsize=(14,6))
    for j in range(8):
        for i,c in enumerate(CONDITIONS):
            axes[i,j].imshow(images[c][j]);axes[i,j].axis('off');axes[i,j].set_title(f'{c}\nlabel {entries[j]["label"]}',fontsize=9)
    fig.suptitle('Fixed first 8 test samples: clean / brightness ×0.5 / Gaussian blur radius 1');fig.tight_layout();save(fig,'synthetic_conditions_examples')
    for v,title in zip(VARIANTS,names):
        rows=read_csv(out/f'{v}_s42/clean/predictions.csv')
        errors=[(i,r) for i,r in enumerate(rows) if r['true']!=r['predicted']]
        errors.sort(key=lambda pair:float(pair[1]['max_softmax']),reverse=True)
        fig,axes=plt.subplots(3,4,figsize=(8,7))
        for ax in axes.flat:ax.axis('off')
        for ax,(i,r) in zip(axes.flat,errors[:12]):
            ax.imshow(images['clean'][i]);ax.set_title(f'True {r["true"]} → pred {r["predicted"]}\nsoftmax {float(r["max_softmax"]):.3f}',fontsize=9)
        fig.suptitle(f'{title}, seed 42: up to 12 highest-softmax clean-test errors');fig.tight_layout();save(fig,f'errors_{v}_seed42')
    return agg

def main(a):
    study,root,out=Path(a.study),Path(a.data),Path(a.out);out.mkdir(parents=True,exist_ok=True)
    if a.device.startswith('cuda'):assert torch.cuda.is_available(),'CUDA unavailable; select GPU or explicitly use --device cpu'
    torch.set_num_threads(2);torch.manual_seed(2026)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    entries,ckhashes=validate(study,root)
    protocol_hash=hashlib.sha256(json.dumps(PROTOCOL,sort_keys=True).encode()).hexdigest()
    save_json(out/'protocol.json',PROTOCOL)
    save_json(out/'environment.json',{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
        'device':a.device,'gpu':torch.cuda.get_device_name() if a.device.startswith('cuda') else None,
        'batch_size':a.batch_size,'workers':a.workers,'source_sha256':{n:sha(Path(__file__).parent/n) for n in ['evaluate_final.py','models.py','data.py']}})
    print('12 checkpoints and fixed manifests validated. No training will run.',flush=True)
    images=preload(root,entries);all_rows=[]
    for v in VARIANTS:
        for seed in SEEDS:
            name=f'{v}_s{seed}';ck=torch.load(study/'runs'/name/'best.pt',map_location='cpu',weights_only=True)
            model=ResNet20(v);model.load_state_dict(ck['model'],strict=True);model.to(a.device)
            for c in CONDITIONS:
                dest=out/name/c;dest.mkdir(parents=True,exist_ok=True);mp=dest/'metrics.json'
                if mp.exists():
                    met=json.loads(mp.read_text())
                    assert met.get('complete') and met['protocol_sha256']==protocol_hash and met['checkpoint_sha256']==ckhashes[name]
                    assert met['manifest_sha256']==HASHES['test'] and met['samples']==12630
                    assert len(read_csv(dest/'predictions.csv'))==12630
                    cm=np.loadtxt(dest/'confusion_matrix.csv',delimiter=',')
                    assert int(cm.sum())==12630
                    print('Reuse completed evaluation:',name,c,flush=True)
                else:
                    met,cm,rows=evaluate_one(model,images[c],entries,a.device,a.batch_size,a.workers)
                    met.update({'complete':True,'split':'test','condition':c,'synthetic_condition':c!='clean',
                        'variant':v,'seed':seed,'best_epoch':ck['epoch'],'checkpoint_sha256':ckhashes[name],
                        'manifest_sha256':HASHES['test'],'protocol_sha256':protocol_hash})
                    write_csv(dest/'predictions.csv',rows);np.savetxt(dest/'confusion_matrix.csv',cm,fmt='%d',delimiter=',')
                    write_csv(dest/'per_class.csv',met['per_class']);save_json(mp,met)
                    print(f'{name} {c}: accuracy={met["accuracy"]*100:.3f}%, Macro-F1={met["macro_f1"]*100:.3f}%',flush=True)
                all_rows.append({k:met[k] for k in ['variant','seed','condition','accuracy','macro_f1','macro_recall','loss','samples','best_epoch','checkpoint_sha256','manifest_sha256']})
            del model
    write_csv(out/'test_per_run.csv',all_rows);agg=figures(out,all_rows,images,entries)
    save_json(out/'test_results.json',{'protocol':PROTOCOL,'per_run':all_rows,'three_seeds':agg})
    save_json(out/'COMPLETE.json',{'completed_evaluations':36,'runs':12,'conditions':3,'samples_per_evaluation':12630,'protocol_sha256':protocol_hash})
    print('Completed all 36 held-out evaluations. Download results before ending the session.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--data',required=True);p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch-size',type=int,default=256);p.add_argument('--workers',type=int,default=2)
    main(p.parse_args())
