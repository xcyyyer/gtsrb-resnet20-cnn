"""Reproduce the fixed four-variant, three-seed experiment in separate outputs."""
import argparse,subprocess,sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data');p.add_argument('--splits',default='study/splits')
    p.add_argument('--out',default='reproduction_runs');p.add_argument('--device',default='cuda')
    a=p.parse_args();project=Path(__file__).resolve().parent
    for seed in [42,10,20]:
        for variant in ['baseline','ds','se','ds_se']:
            out=Path(a.out)/f'{variant}_s{seed}'
            args=[sys.executable,str(project/'run.py'),'train','--root',str(Path(a.data).resolve()),
                '--splits',str(Path(a.splits).resolve()),'--out',str(out.resolve()),'--variant',variant,
                '--seed',str(seed),'--epochs','80','--batch-size','128','--workers','2','--lr','0.05',
                '--size','32','--device',a.device]
            if (out/'last.pt').is_file():args+=['--resume','--allow-relocation']
            print('Run:',variant,'seed',seed,flush=True)
            subprocess.run(args,check=True,cwd=project)

if __name__=='__main__':main()
