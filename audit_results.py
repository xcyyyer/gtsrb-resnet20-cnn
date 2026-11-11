"""Audit real GTSRB training logs and checkpoints; no held-out evaluation."""
import argparse, csv, hashlib, json, math, statistics
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

VARIANTS = ['baseline', 'ds', 'se', 'ds_se']
SEEDS = [42, 10, 20]
COLORS = dict(zip(VARIANTS, ['#2456A6', '#168477', '#D18A21', '#8956A3']))
NAMES = {'baseline':'ResNet-20', 'ds':'DS', 'se':'SE', 'ds_se':'DS + SE'}

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read_csv(p):
    with open(p, newline='', encoding='utf-8-sig') as f: return list(csv.DictReader(f))
def write_csv(p, rows):
    with open(p, 'w', newline='', encoding='utf-8-sig') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
def save(fig, out, name):
    fig.savefig(out/(name+'.png'), dpi=200, bbox_inches='tight')
    fig.savefig(out/(name+'.pdf'), bbox_inches='tight'); plt.close(fig)

def main(study, out):
    study, out = Path(study), Path(out); out.mkdir(parents=True, exist_ok=True)
    figs=out/'figures'; figs.mkdir(exist_ok=True)
    summary=json.loads((study/'splits/split_summary.json').read_text())
    hashes={s:sha(study/'splits'/f'{s}.csv') for s in ['train','val','test']}
    assert hashes==summary['manifest_sha256'], 'Manifest SHA256 mismatch'
    splits={s:read_csv(study/'splits'/f'{s}.csv') for s in hashes}
    assert {s:len(r) for s,r in splits.items()}==summary['counts']
    for s, rows in splits.items():
        assert len({r['path'] for r in rows})==len(rows), 'Repeated image paths'
        assert {int(r['label']) for r in rows}==set(range(43))
    overlap=set(r['group'] for r in splits['train']) & set(r['group'] for r in splits['val'])
    assert not overlap, 'Train/validation track overlap'
    for a,b in [('train','val'),('train','test'),('val','test')]:
        assert not ({r['path'] for r in splits[a]} & {r['path'] for r in splits[b]})
    import torch
    from models import ResNet20, profile
    torch.set_num_threads(2)
    rows=[]; histories={}; checkpoint_checks=[]
    for v in VARIANTS:
        computed_profile=profile(ResNet20(v),32)
        for seed in SEEDS:
            run=study/'runs'/f'{v}_s{seed}'
            hist=read_csv(run/'history.csv'); histories[v,seed]=hist
            assert [int(r['epoch']) for r in hist]==list(range(1,81))
            for r in hist:
                assert all(math.isfinite(float(x)) for x in r.values())
                assert abs(float(r['lr'])-.05*(1+math.cos(math.pi*(int(r['epoch'])-1)/80))/2)<1e-10
            best=max(hist,key=lambda r:float(r['val_accuracy'])) # first maximum = checkpoint policy
            conf=json.loads((run/'config.json').read_text())
            assert (conf['variant'],conf['seed'],conf['epochs'],conf['size'],conf['batch_size'],conf['lr'])==(v,seed,80,32,128,.05)
            assert (run/'last.pt').is_file()
            vs=json.loads((run/'validation_summary.json').read_text())
            assert vs['completed_epochs']==80
            assert abs(vs['best_val_accuracy']-float(best['val_accuracy']))<1e-12
            stored_profile=json.loads((run/'profile.json').read_text())
            assert stored_profile==computed_profile
            env=json.loads((run/'environment.json').read_text())
            assert env['train_val_hashes']=={s:hashes[s] for s in ['train','val']}
            for name in ['models.py','data.py']:
                assert sha(Path(__file__).parent/name)==env['source_sha256'][name]
            ck=torch.load(run/'best.pt',map_location='cpu',weights_only=True)
            assert ck['epoch']==int(best['epoch'])
            assert abs(ck['val_metrics']['accuracy']-float(best['val_accuracy']))<1e-12
            assert abs(ck['val_metrics']['macro_f1']-float(best['val_macro_f1']))<1e-12
            assert ck['manifest_hashes']=={s:hashes[s] for s in ['train','val']}
            assert (ck['config']['variant'],ck['config']['seed'])==(v,seed)
            model=ResNet20(v); model.load_state_dict(ck['model'],strict=True); model.eval()
            with torch.inference_mode(): y=model(torch.zeros(2,3,32,32))
            assert y.shape==(2,43) and torch.isfinite(y).all()
            checkpoint_checks.append({'run':run.name,'best_epoch':ck['epoch'], 'best_sha256':sha(run/'best.pt'),
                'last_sha256':sha(run/'last.pt'),'strict_load':True,'finite_forward':True})
            rows.append({'run':run.name,'variant':v,'seed':seed,'epochs':80,'best_epoch':int(best['epoch']),
                'best_val_accuracy_pct':100*float(best['val_accuracy']),
                'val_macro_f1_at_best_accuracy_pct':100*float(best['val_macro_f1']),
                'parameters':stored_profile['parameters'],'conv_linear_macs':stored_profile['conv_linear_macs'],
                'logged_epoch_seconds_sum':sum(float(r['epoch_seconds']) for r in hist),
                'migrated':bool(list(run.glob('migration_*.json')))})
    aggregated=[]
    for v in VARIANTS:
        vr=[r for r in rows if r['variant']==v]
        aggregated.append({'variant':v,'n_seeds':3,
            'val_accuracy_mean_pct':statistics.mean(r['best_val_accuracy_pct'] for r in vr),
            'val_accuracy_sample_std_pp':statistics.stdev(r['best_val_accuracy_pct'] for r in vr),
            'val_macro_f1_mean_pct':statistics.mean(r['val_macro_f1_at_best_accuracy_pct'] for r in vr),
            'val_macro_f1_sample_std_pp':statistics.stdev(r['val_macro_f1_at_best_accuracy_pct'] for r in vr),
            'parameters':vr[0]['parameters'],'conv_linear_macs':vr[0]['conv_linear_macs']})
    audit={'runs':12,'epochs_per_run':80,'total_logged_epochs':960,'counts':summary['counts'],
        'train_val_track_overlap':0,'manifest_sha256':hashes,'checkpoint_checks':checkpoint_checks,
        'torch_for_audit':torch.__version__,'test_evaluation_present':(out.parent/'test_evaluation/COMPLETE.json').is_file(),
        'warning':'Source results.json/results.csv are stale (8 entries); rebuilt from all 12 histories.',
        'limitations':['Raw images are obtained separately; image-level near-duplicate search was not performed.',
            'A finite zero-input forward pass verifies weight loading; held-out results are stored separately in test_evaluation.',
            'DS+SE seed20 resumed across platforms; logged times are not a controlled speed benchmark.']}
    write_csv(out/'validation_per_run.csv',rows); write_csv(out/'validation_three_seeds.csv',aggregated)
    (out/'audit.json').write_text(json.dumps(audit,indent=2))
    (out/'validation_results.json').write_text(json.dumps({'per_run':rows,'three_seeds':aggregated},indent=2))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    for v in VARIANTS:
        for seed in SEEDS:
            hist=histories[v,seed]; epochs=np.arange(1,81)
            b=next(r for r in rows if r['variant']==v and r['seed']==seed)
            fig,axes=plt.subplots(1,3,figsize=(14,3.8))
            for prefix,label,color in [('train','Train','#2456A6'),('val','Validation','#DC842F')]:
                axes[0].plot(epochs,[float(r[prefix+'_loss']) for r in hist],label=label,color=color)
                axes[1].plot(epochs,100*np.array([float(r[prefix+'_accuracy']) for r in hist]),label=label,color=color)
            axes[2].plot(epochs,100*np.array([float(r['val_macro_f1']) for r in hist]),color='#168477')
            axes[0].set(title='Cross-entropy loss',ylabel='Loss')
            axes[1].set(title='Accuracy',ylabel='Accuracy (%)')
            axes[2].set(title='Validation Macro-F1',ylabel='Macro-F1 (%)')
            for ax in axes:
                ax.set_xlabel('Epoch'); ax.grid(alpha=.18)
                ax.axvline(b['best_epoch'],color='gray',lw=.8,ls='--')
            if b['migrated']:
                for ax in axes: ax.axvline(39.5,color='#8956A3',lw=1,ls=':')
            axes[0].legend(); axes[1].legend()
            fig.suptitle(f'{NAMES[v]} | seed {seed} | best val accuracy {b["best_val_accuracy_pct"]:.3f}% at epoch {b["best_epoch"]}')
            fig.tight_layout(); save(fig,figs,f'curves_{v}_s{seed}')
    plt.rcParams.update({'font.size':14})
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for v in VARIANTS:
        for ax,metric,scale in [(axes[0,0],'train_loss',1),(axes[0,1],'val_loss',1),
                                (axes[1,0],'val_accuracy',100),(axes[1,1],'val_macro_f1',100)]:
            arr=np.array([[float(r[metric])*scale for r in histories[v,s]] for s in SEEDS])
            mean=arr.mean(0); sd=arr.std(0,ddof=1)
            ax.plot(np.arange(1,81),mean,color=COLORS[v],label=NAMES[v])
            ax.fill_between(np.arange(1,81),mean-sd,mean+sd,color=COLORS[v],alpha=.12)
    for ax,title,ylabel in zip(axes.flat,['Training loss','Validation loss','Validation accuracy','Validation Macro-F1'],['Loss','Loss','Accuracy (%)','Macro-F1 (%)']):
        ax.set(title=title,xlabel='Epoch',ylabel=ylabel); ax.grid(alpha=.18); ax.legend()
    fig.suptitle('Seeds 42, 10, 20: mean and sample standard deviation (shaded)');fig.tight_layout()
    save(fig,figs,'three_seed_learning_curves')
    plt.rcParams.update({'font.size':10})
    fig,axes=plt.subplots(1,2,figsize=(11,4.3))
    for ax,m,s,title in [(axes[0],'val_accuracy_mean_pct','val_accuracy_sample_std_pp','Best validation accuracy'),
                         (axes[1],'val_macro_f1_mean_pct','val_macro_f1_sample_std_pp','Macro-F1 at selected checkpoint')]:
        for i,r in enumerate(aggregated):
            ax.errorbar(i,r[m],yerr=r[s],fmt='o',color=COLORS[r['variant']],capsize=5,ms=8)
            ax.annotate(f'{r[m]:.3f} ± {r[s]:.3f}',(i,r[m]+r[s]),xytext=(0,7),textcoords='offset points',ha='center',fontsize=9)
        ax.set_xticks(range(4),[NAMES[v] for v in VARIANTS]);ax.set(title=title,ylabel='Percent (%)',xlim=(-.65,3.65));ax.grid(axis='y',alpha=.18);ax.margins(y=.35)
    fig.suptitle('Mean ± sample SD over 3 training seeds; validation only');fig.tight_layout()
    save(fig,figs,'validation_mean_std')
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,k,scale,title in [(axes[0],'parameters',1e3,'Trainable parameters (thousands)'),(axes[1],'conv_linear_macs',1e6,'Conv/Linear MACs (millions)')]:
        values=[r[k]/scale for r in aggregated]
        ax.bar(range(4),values,color=[COLORS[v] for v in VARIANTS])
        ax.set_xticks(range(4),[NAMES[v] for v in VARIANTS]);ax.set_title(title);ax.set_ylim(0,max(values)*1.18)
        for i,value in enumerate(values): ax.text(i,value,f'{value:.3f}',ha='center',va='bottom')
    fig.suptitle('Input: 1 × 3 × 32 × 32; 1 MAC = one multiply-accumulate');fig.tight_layout()
    save(fig,figs,'model_complexity')
    fig,ax=plt.subplots(figsize=(7.5,5))
    for r in aggregated:
        ax.errorbar(r['conv_linear_macs']/1e6,r['val_accuracy_mean_pct'],yerr=r['val_accuracy_sample_std_pp'],fmt='o',capsize=4,color=COLORS[r['variant']],label=NAMES[r['variant']],ms=8)
    ax.set(xlabel='Conv/Linear MACs (millions)',ylabel='Best validation accuracy (%)',title='Computation–accuracy trade-off: mean ± sample SD')
    ax.grid(alpha=.2);ax.legend();fig.tight_layout();save(fig,figs,'compute_accuracy_tradeoff')
    # Fixed validation protocol; no statistical-significance claim for three seeds.
    base=aggregated[0]; ds=aggregated[1]
    param_reduction=100*(1-ds['parameters']/base['parameters'])
    mac_reduction=100*(1-ds['conv_linear_macs']/base['conv_linear_macs'])
    md=['# GTSRB 训练结果核验（仅验证集）','',
        '来源：用户提供的 GTSRB_after_migration.zip。本页仅汇总验证结果；正式测试输出另见test_evaluation，不得把本页数字写为测试结果。','',
        '## 核验结论','',
        '- 4 种结构 × 种子 42、10、20，共 12 组；每组日志连续记录 1–80 轮，共 960 轮。',
        '- 12 个 best.pt 均成功严格加载，输出形状为 2×43，前向输出有限；best.pt 中轮次、验证准确率和 Macro-F1 与日志所选轮次一致。',
        '- 训练 31,379；验证 7,830；测试清单 12,630。三份划分文件 SHA256 与记录一致，训练/验证轨迹组交集为 0。',
        '- models.py/data.py 与训练环境记录的源码校验值一致；参数量与 Conv/Linear MACs 已由模型代码重新计算核对。',
        '- 原 runs/results.json 只含 8 组，并且有旧数值；本次直接从 12 份 history.csv 重建汇总，原始文件未改动。','',
        '## 三种种子汇总','',
        '各组按验证准确率最大值选择检查点，并列取首次出现者；Macro-F1取同一轮。标准差为样本标准差（ddof=1）；准确率/F1均为百分数。','',
        '| 结构 | 验证 Accuracy（均值±SD，%） | 验证 Macro-F1（均值±SD，%） | 参数量 | Conv/Linear MACs |',
        '|---|---:|---:|---:|---:|']
    for r in aggregated:
        md.append(f'| {r["variant"]} | {r["val_accuracy_mean_pct"]:.3f} ± {r["val_accuracy_sample_std_pp"]:.3f} | {r["val_macro_f1_mean_pct"]:.3f} ± {r["val_macro_f1_sample_std_pp"]:.3f} | {r["parameters"]:,} | {r["conv_linear_macs"]:,} |')
    md+=['','## 可据实写入的分析','',
        f'- DS 相对基线参数减少 {param_reduction:.3f}%，Conv/Linear MACs 减少 {mac_reduction:.3f}%；最佳验证准确率的三种子均值差为 {ds["val_accuracy_mean_pct"]-base["val_accuracy_mean_pct"]:+.3f} 个百分点。',
        '- 因此目前应强调结构轻量化与验证准确率接近；不能沿用单种子“DS提高精度”的结论，也不能把3次结果描述为统计显著。',
        '- SE 与 DS+SE 的平均验证准确率均未超过基线。这是需要保留和解释的负结果。可能原因只能作为假设，不能据此断言注意力机制无效。',
        '- ds_se_s20 在第39轮后由 Colab 迁移至 Kaggle，配置仅改变路径；保留迁移记录。该组训练时长混合不同平台，不用于模型速度排名。',
        '- MACs仅统计卷积与全连接乘加，不含BN、激活、池化、残差加法及SE逐元素乘法；不能直接标为完整FLOPs或推断实际加速倍数。','',
        '## 图与文件','',
        '- figures/curves_*：12组独立 Loss、Accuracy、验证Macro-F1 曲线（PNG+PDF）。',
        '- figures/three_seed_learning_curves：四种模型三种子的均值与样本标准差曲线。',
        '- figures/validation_mean_std：所选检查点验证指标误差棒图。',
        '- figures/model_complexity、compute_accuracy_tradeoff：模型复杂度与验证精度权衡。',
        '- validation_per_run.csv、validation_three_seeds.csv、validation_results.json：保留全精度数据；显示值统一三位小数。',
        '- audit.json：检查点文件校验值及核验范围。','',
        '## 正式测试与交付','',
        '用各组已选定的 best.pt，统一评估官方测试集 clean/dark/blur。dark=图像缩放至32×32后亮度×0.5；blur=缩放后GaussianBlur(radius=1)。两种扰动属于合成场景，不代表真实夜间或实际摄像头性能。',
        '正式测试的均值/标准差、混淆矩阵、逐类指标和错误样例请见test_evaluation；代码仓库线上发布仍待完成。']
    (out/'核验与验证结果说明.md').write_text('\n'.join(md),encoding='utf-8')
    print(json.dumps({'runs':12,'checks':'passed','three_seeds':aggregated},ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--study',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();main(a.study,a.out)
