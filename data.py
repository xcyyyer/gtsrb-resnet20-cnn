"""Official final GTSRB archives; group-stratified train/val split by track."""
import argparse
import csv
import hashlib
import json
import random
import urllib.request
import urllib.error
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import Dataset

BASE = 'https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/'
ARCHIVES = {'GTSRB_Final_Training_Images.zip': 'f33fd80ac59bff73c82d25ab499e03a3',
            'GTSRB_Final_Test_Images.zip': 'c7e4e6327067d32654124b0fe9e82185',
            'GTSRB_Final_Test_GT.zip': 'fe31e9c9270bbcd7b84b7f21a9d9d9e5'}
# Archive MD5: datasets-mila/datasets--gtsrb/md5sums; record SHA256 locally too.

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''): h.update(chunk)
    return h.hexdigest()

def download_official(root):
    root = Path(root); root.mkdir(parents=True, exist_ok=True)
    records = []
    for filename, md5 in ARCHIVES.items():
        dest = root / filename
        if not dest.exists():
            tmp = dest.with_suffix('.part')
            print('Downloading', filename, flush=True)
            with urllib.request.urlopen(BASE + filename, timeout=60) as src, open(tmp, 'wb') as out:
                while chunk := src.read(1024*1024): out.write(chunk)
            tmp.replace(dest)
        if md5:
            h = hashlib.md5(dest.read_bytes()).hexdigest()
            if h != md5: raise ValueError(f'Checksum mismatch: {dest}; remove and download again')
        digest = sha256(dest)
        marker = root / (filename + '.extracted')
        if not marker.exists() or marker.read_text() != digest:
            with zipfile.ZipFile(dest) as z:
                if z.testzip(): raise ValueError('Archive CRC failure: ' + filename)
                for name in z.namelist():
                    if not (root / name).resolve().is_relative_to(root.resolve()):
                        raise ValueError('Unsafe archive member')
                z.extractall(root)
            marker.write_text(digest)
        records.append({'file': filename, 'url': BASE+filename, 'sha256': digest})
    (root / 'archive_provenance.json').write_text(json.dumps(records, indent=2))

MIRROR = 'https://www.kaggle.com/api/v1/datasets/download/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign'

def download(root, source="auto"):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    if (root/'Train.csv').exists():
        print('Using existing Kaggle mirror extraction'); return
    if source != 'kaggle':
        try:
            download_official(root); return
        except urllib.error.URLError as e:
            if source == 'official': raise
            print(f'Official download unavailable ({e}); trying public Kaggle mirror.',flush=True)
    dest=root/'kaggle_gtsrb.zip'
    if not dest.exists():
        tmp=dest.with_suffix('.part')
        with urllib.request.urlopen(MIRROR,timeout=60) as src, open(tmp,'wb') as out:
            while chunk:=src.read(1024*1024): out.write(chunk)
        tmp.replace(dest)
    with zipfile.ZipFile(dest) as z:
        if z.testzip(): raise ValueError('Kaggle archive CRC failure')
        for name in z.namelist():
            if not (root/name).resolve().is_relative_to(root.resolve()): raise ValueError('Unsafe archive member')
        z.extractall(root)
    if not (root/'Train.csv').exists(): raise FileNotFoundError('Mirror structure changed: Train.csv missing')
    (root/'archive_provenance.json').write_text(json.dumps({'source':'Kaggle mirror',
        'url':MIRROR,'archive_sha256':sha256(dest),
        'note':'Repackaged PNG images and CSVs. Counts and label/track mapping verified by prepare.'},indent=2))

def write_csv(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['path', 'label', 'group']); w.writeheader(); w.writerows(rows)

def prepare(root, out, seed=2026, val_fraction=0.2):
    root, out = Path(root).resolve(), Path(out)
    train_dir = root / 'GTSRB/Final_Training/Images'
    mirror = (root/'Train.csv').exists()
    if mirror:
        with open(root/'Train.csv',newline='') as f: train_meta=list(csv.DictReader(f))
        files=sorted(root/row['Path'] for row in train_meta)
        for row in train_meta:
            assert int(Path(row['Path']).parent.name)==int(row['ClassId'])
    else:
        files = sorted(train_dir.glob('*/*.ppm'))
    if len(files) != 39209: raise ValueError(f'Expected FINAL training set 39209, found {len(files)} at {train_dir}')
    groups = defaultdict(lambda: defaultdict(list))
    for p in files:
        label = int(p.parent.name)
        track = p.stem.rsplit('_',1)[0]
        if not p.is_file(): raise FileNotFoundError(p)
        group = f'{label:05d}/{track}'
        groups[label][group].append({'path': str(p.relative_to(root)), 'label': label, 'group': group})
    if set(groups) != set(range(43)): raise ValueError('Expected classes 0..42')
    train, val = [], []
    rng = random.Random(seed)
    for label in sorted(groups):
        keys = sorted(groups[label]); rng.shuffle(keys)
        n = max(1, min(len(keys)-1, round(len(keys)*val_fraction)))
        for i, g in enumerate(keys): (val if i < n else train).extend(groups[label][g])
    assert not ({x['group'] for x in train} & {x['group'] for x in val})
    test = []
    if mirror:
        with open(root/'Test.csv',newline='') as f:
            for row in csv.DictReader(f):
                p=root/row['Path']
                if not p.is_file(): raise FileNotFoundError(p)
                test.append({'path':str(p.relative_to(root)),'label':int(row['ClassId']),'group':'official_test'})
    else:
        gt = list(root.rglob('GT-final_test.csv'))
        if len(gt) != 1: raise ValueError(f'Expected one GT-final_test.csv, got {gt}')
        with open(gt[0], newline='') as f:
            for row in csv.DictReader(f, delimiter=';'):
                p = root / 'GTSRB/Final_Test/Images' / row['Filename']
                if not p.is_file(): raise FileNotFoundError(p)
                test.append({'path': str(p.relative_to(root)), 'label': int(row['ClassId']), 'group': 'official_test'})
    assert len(test) == 12630
    if out.exists() and any(out.iterdir()): raise FileExistsError(f'{out} exists; reuse it, or choose a new output directory')
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in [('train', train), ('val', val), ('test', test)]: write_csv(out / f'{name}.csv', rows)
    summary = {'source_layout': 'kaggle_png' if mirror else 'official_ppm', 'seed': seed, 'requested_val_group_fraction': val_fraction,
               'counts': {k: len(v) for k,v in [('train',train),('val',val),('test',test)]},
               'class_counts': {k: dict(Counter(int(r['label']) for r in v)) for k,v in [('train',train),('val',val),('test',test)]},
               'group_overlap_train_val': 0,
               'manifest_sha256': {k: sha256(out / f'{k}.csv') for k in ['train','val','test']}}
    (out / 'split_summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return summary

class Signs(Dataset):
    def __init__(self, root, manifest, size=32, augment=False, condition='clean'):
        self.root, self.size, self.augment, self.condition = Path(root), size, augment, condition
        with open(manifest, newline='') as f: self.rows = list(csv.DictReader(f))
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row['path']) as src: im = src.convert('RGB')
        im = im.resize((self.size,self.size), Image.Resampling.BILINEAR)
        if self.augment:
            # Flips forbidden: left/right arrows and digits can change meaning.
            im = im.rotate(random.uniform(-10,10), resample=Image.Resampling.BILINEAR)
            im = ImageEnhance.Brightness(im).enhance(random.uniform(.8,1.2))
            im = ImageEnhance.Contrast(im).enhance(random.uniform(.8,1.2))
        if self.condition == 'dark': im = ImageEnhance.Brightness(im).enhance(.5)
        if self.condition == 'blur': im = im.filter(ImageFilter.GaussianBlur(1.0))
        x = torch.from_numpy(np.array(im, dtype=np.float32).transpose(2,0,1).copy()) / 255.
        return (x-.5)/.5, int(row['label']), row['path']

if __name__ == '__main__':
    p=argparse.ArgumentParser(); p.add_argument('action',choices=['download','prepare']); p.add_argument('--root',default='data'); p.add_argument('--out',default='splits'); p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--source', choices=['auto','official','kaggle'],default='auto')
    a=p.parse_args()
    if a.action=='download': download(a.root,a.source)
    else: prepare(a.root,a.out,a.seed)
