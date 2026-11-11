"""Classify one cropped traffic-sign image; numeric IDs follow GTSRB."""
import argparse,json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from models import ResNet20

p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--image',required=True)
p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu'); a=p.parse_args()
c=torch.load(a.checkpoint,map_location='cpu',weights_only=False); conf=c['config']
m=ResNet20(conf['variant']); m.load_state_dict(c['model']); m.to(a.device).eval()
with Image.open(a.image) as f: im=f.convert('RGB').resize((conf['size'],conf['size']),Image.Resampling.BILINEAR)
x=torch.from_numpy(np.array(im,dtype=np.float32).transpose(2,0,1).copy())/255.
with torch.inference_mode(): values,ids=m(((x-.5)/.5).unsqueeze(0).to(a.device)).softmax(1)[0].topk(5)
print(json.dumps([{'class_id':int(i),'softmax_score':float(v)} for i,v in zip(ids,values)],indent=2))
