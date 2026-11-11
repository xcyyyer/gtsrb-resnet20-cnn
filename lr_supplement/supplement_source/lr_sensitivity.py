"""Validation-only, post-hoc learning-rate sensitivity study.

This supplements the frozen four-architecture study. It never evaluates test
images and never replaces the existing official-test conclusions.
"""
from pathlib import Path
import argparse, csv, hashlib, json, os, platform, subprocess, sys, time, zipfile

HERE=Path(__file__).resolve().parent
LRS=(0.01,0.05,0.1)
EPOCHS=80
SEED=42
EXPECTED_COUNTS={'train':31379,'val':7830}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,p)
def run_name(lr):return 'ds_lr'+format(lr,'g').replace('.','p')+'_s42'
def read_history(folder):
    p=Path(folder)/'history.csv'
    if not p.exists():return []
    with p.open(newline='',encoding='utf-8-sig') as f:rows=list(csv.DictReader(f))
    if [int(x['epoch']) for x in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('日志轮次缺失或重复：'+str(p))
    if len(rows)>EPOCHS:raise ValueError('训练轮数超过固定预算')
    return rows

def validate_sources():
    pins=json.loads((HERE/'SOURCE_SHA256.json').read_text())
    for name,digest in pins.items():
        if sha(HERE/name)!=digest:raise ValueError('源码发生变化：'+name)
    return pins

def validate_data(root):
    root=Path(root).resolve()
    pins=json.loads((HERE/'SPLIT_SHA256.json').read_text())
    with (root/'Train.csv').open(newline='',encoding='utf-8-sig') as f:
        metadata={r['Path'].replace('\\','/'):int(r['ClassId']) for r in csv.DictReader(f)}
    if len(metadata)!=39209:raise ValueError('Train.csv不是原版39,209张训练来源')
    groups={};paths=set()
    for split,n in EXPECTED_COUNTS.items():
        manifest=HERE/'splits'/f'{split}.csv'
        if sha(manifest)!=pins[split]:raise ValueError('固定划分校验失败：'+split)
        with manifest.open(newline='',encoding='utf-8') as f:rows=list(csv.DictReader(f))
        if len(rows)!=n:raise ValueError('样本量不匹配：'+split)
        groups[split]=set()
        for r in rows:
            rel=r['path'].replace('\\','/')
            if rel in paths:raise ValueError('发现重复路径：'+rel)
            paths.add(rel);groups[split].add(r['group'])
            if metadata.get(rel)!=int(r['label']):raise ValueError('训练标签或路径不一致：'+rel)
            image=(root/rel).resolve()
            if not image.is_relative_to(root) or not image.is_file():raise FileNotFoundError(image)
    if groups['train'] & groups['val']:raise ValueError('训练与验证轨迹组重叠')
    return pins

def runtime(device):
    import numpy as np
    import torch
    if device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('当前没有可用GPU，请先启用GPU。本实验不自动转到CPU训练。')
    return {'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,
            'device':device,'gpu':torch.cuda.get_device_name(0) if device=='cuda' else None,
            'cuda_runtime':torch.version.cuda,'cudnn':torch.backends.cudnn.version()}

def protocol(out,env,source,manifest):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    fixed={'study':'post_hoc_validation_lr_sensitivity_v1','variant':'ds','seed':SEED,
           'initial_lrs':list(LRS),'epochs':EPOCHS,'batch_size':128,'workers':2,'size':32,
           'optimizer':'SGD','momentum':0.9,'weight_decay':0.0005,
           'scheduler':'CosineAnnealingLR T_max=80','source_sha256':source,
           'manifest_sha256':manifest,'environment':env,
           'checkpoint_rule':'highest validation accuracy; first epoch on ties',
           'comparison_rule':'highest best validation accuracy; lower initial LR on ties',
           'scope':'Single seed descriptive sensitivity experiment; no significance claim.',
           'test_policy':'Original test results were already inspected. No test evaluation, tuning, or replacement of original conclusions in this supplementary study.'}
    path=out/'protocol.json'
    if path.exists():
        if json.loads(path.read_text())!=fixed:
            raise ValueError('续训环境、源码或实验设置发生变化。请恢复同环境，或使用新的输出目录，不混合两套实验。')
    else:write_json(path,fixed)
    return fixed

def extract_metrics(folder,lr,manifest):
    import torch
    folder=Path(folder);rows=read_history(folder)
    if len(rows)!=EPOCHS:raise ValueError('该组未完成80轮：'+str(folder))
    config=json.loads((folder/'config.json').read_text())
    expected={'variant':'ds','seed':SEED,'epochs':EPOCHS,'batch_size':128,'lr':lr,'size':32,'workers':2}
    for k,v in expected.items():
        if config.get(k)!=v:raise ValueError('该组配置不符合预设：'+k)
    best=max(rows,key=lambda r:float(r['val_accuracy']))
    c=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    if c['manifest_hashes']!=manifest:raise ValueError('最佳权重的数据划分不匹配')
    if c['epoch']!=int(best['epoch']):raise ValueError('最佳权重轮次与日志不符')
    for key,logkey in [('accuracy','val_accuracy'),('macro_f1','val_macro_f1')]:
        if abs(float(c['val_metrics'][key])-float(best[logkey]))>1e-12:
            raise ValueError('最佳权重指标与日志不符：'+key)
    return {'initial_lr':lr,'seed':SEED,'completed_epochs':len(rows),'best_epoch':int(best['epoch']),
            'best_val_accuracy_pct':100*float(best['val_accuracy']),
            'val_macro_f1_at_best_accuracy_pct':100*float(best['val_macro_f1']),
            'val_loss_at_best_accuracy':float(best['val_loss']),
            'last_val_accuracy_pct':100*float(rows[-1]['val_accuracy']),
            'logged_train_minutes':sum(float(r['epoch_seconds']) for r in rows)/60,
            'checkpoint_sha256':sha(folder/'best.pt'),'history_sha256':sha(folder/'history.csv')}

def summarize(out):
    out=Path(out);spec=json.loads((out/'protocol.json').read_text())
    rows=[];status=[]
    for lr in LRS:
        folder=out/'runs'/run_name(lr);hist=read_history(folder)
        status.append({'initial_lr':lr,'completed_epochs':len(hist)})
        if len(hist)==EPOCHS:rows.append(extract_metrics(folder,lr,spec['manifest_sha256']))
    write_json(out/'progress.json',status)
    if rows:
        with (out/'validation_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    complete=len(rows)==len(LRS)
    winner=sorted(rows,key=lambda r:(-r['best_val_accuracy_pct'],r['initial_lr']))[0] if complete else None
    result={'complete':complete,'per_lr':rows,'validation_selected_lr':winner['initial_lr'] if winner else None,
            'note':'真实训练输出。单种子验证集补充实验，不替换原四结构测试结果，不代表跨种子的稳定最优值。'}
    write_json(out/'validation_summary.json',result)
    if complete:
        plot(out,rows)
        text=['# 验证集学习率敏感性结果','',
              '本实验在已完成原研究后追加。仅使用原训练集和验证集，未运行任何测试集评估。',
              '固定DS、种子42、每组80轮，只改变初始学习率；三个值都在同一环境独立训练。','',
              '| 初始学习率 | 最佳轮次 | 验证Accuracy/% | 同轮Macro-F1/% | 同轮验证Loss |',
              '|---|---:|---:|---:|---:|']
        for r in rows:text.append(f"| {r['initial_lr']:g} | {r['best_epoch']} | {r['best_val_accuracy_pct']:.3f} | {r['val_macro_f1_at_best_accuracy_pct']:.3f} | {r['val_loss_at_best_accuracy']:.6f} |")
        text+=['',f"按预设验证Accuracy规则，此次种子42下选择的初始学习率为{winner['initial_lr']:g}。",
               '这是一组学习率敏感性证据，不能声称三个种子或其他场景下一定最优。',
               '结果不论提高或下降均应如实保留；图中不预设高学习率一定震荡或低学习率一定欠拟合。']
        (out/'报告补充结果.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    return result

def plot(out,rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder=out/'figures';folder.mkdir(exist_ok=True)
    fig,axs=plt.subplots(2,2,figsize=(11,7))
    for lr in LRS:
        h=read_history(out/'runs'/run_name(lr));x=[int(r['epoch']) for r in h]
        for ax,key,title in zip(axs.flat,['train_loss','val_loss','train_accuracy','val_accuracy'],
                               ['Training loss','Validation loss','Training accuracy','Validation accuracy']):
            y=[float(r[key])*(100 if 'accuracy' in key else 1) for r in h]
            ax.plot(x,y,label=f'initial LR={lr:g}');ax.set_title(title);ax.set_xlabel('Epoch');ax.grid(alpha=.2)
            ax.set_ylabel('Percent' if 'accuracy' in key else 'Cross entropy')
    for ax in axs.flat:ax.legend()
    fig.suptitle('DS, seed 42, same environment and 80-epoch budget; validation only')
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'lr_learning_curves.{ext}',dpi=180)
    plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(9,3.8))
    for ax,key,title in zip(axs,['best_val_accuracy_pct','val_macro_f1_at_best_accuracy_pct'],
                             ['Best validation accuracy','Macro-F1 at selected epoch']):
        vals=[r[key] for r in rows];ax.bar([str(r['initial_lr']) for r in rows],vals,color=['#3966A2','#268A7D','#CD922F'])
        ax.set_xlabel('Initial learning rate');ax.set_ylabel('Percent');ax.set_title(title)
        ax.set_ylim(max(0,min(vals)-.5),min(100,max(vals)+.25))
        for i,v in enumerate(vals):ax.text(i,v,f'{v:.3f}',ha='center',va='bottom',fontsize=9)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(folder/f'lr_validation_comparison.{ext}',dpi=180)
    plt.close(fig)

def export(out):
    out=Path(out);dest=out.parent/'GTSRB_LR_results.zip';tmp=dest.with_suffix('.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file() and f.suffix!='.tmp':z.write(f,Path(out.name)/f.relative_to(out))
        for f in [HERE/n for n in ['lr_sensitivity.py','run.py','models.py','data.py','migration_config.py','SOURCE_SHA256.json','SPLIT_SHA256.json']]:
            z.write(f,Path('supplement_source')/f.name)
    os.replace(tmp,dest);print('结果备份：'+str(dest),flush=True)
    return dest

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    ap.add_argument('--summarize-only',action='store_true');ap.add_argument('--prepare-only',action='store_true')
    a=ap.parse_args();out=Path(a.out).resolve();root=Path(a.root).resolve()
    if a.summarize_only:summarize(out);export(out);return
    source=validate_sources();manifest=validate_data(root);env=runtime(a.device)
    protocol(out,env,source,manifest)
    print('固定实验：DS，种子42，初始学习率0.01/0.05/0.1，各80轮；禁止使用测试集选择。',flush=True)
    if a.prepare_only:return
    # All three initial learning rates start from scratch; only each run's own
    # last checkpoint may be used when rerunning the same notebook.
    try:
        for lr in LRS:
            folder=out/'runs'/run_name(lr);history=read_history(folder)
            if len(history)==EPOCHS:
                extract_metrics(folder,lr,manifest);print('已完成，跳过：'+folder.name,flush=True);continue
            args=[sys.executable,'-u',str(HERE/'run.py'),'train','--root',str(root),
                  '--splits',str(HERE/'splits'),'--out',str(folder),'--variant','ds','--seed','42',
                  '--epochs','80','--batch-size','128','--lr',str(lr),'--size','32','--workers','2','--device',a.device]
            if (folder/'last.pt').exists():args+=['--resume','--allow-relocation']
            elif folder.exists() and any(folder.iterdir()):
                raise RuntimeError('输出目录已有文件但没有last.pt，不能安全自动续训：'+str(folder))
            print(f'开始 {folder.name}，已有{len(history)}/80轮',flush=True)
            subprocess.run(args,check=True,cwd=HERE)
            summarize(out);export(out)
    finally:
        if (out/'protocol.json').exists():
            # A hard platform shutdown cannot guarantee this finally executes;
            # persistent Drive storage or an exported backup is still needed.
            summarize(out);export(out)

if __name__=='__main__':main()
