import csv, random, statistics, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch.nn.functional as F
import sys; sys.path.insert(0,'.')
from core.ranking.dinov2_ranker import DINOv2Ranker
random.seed(0)
D='evaluation/ver_new/dataset_out'
fd={}
for p in Path('gannhanocr-fd').rglob('U+*.png'):
    try: fd[chr(int(p.stem.replace('U+',''),16))]=str(p)
    except: pass
r=DINOv2Ranker(font_path='font_diffusion/fonts/NomNaTong-Regular.ttf',embedding_cache_dir=f'{D}/emb_cache')
def emb_img(im): return r._embed(im.convert('RGB'))
def cos(a,b): return max(0.0,(float(F.cosine_similarity(a.unsqueeze(0),b.unsqueeze(0)))+1)/2)
def aug(im):
    im=im.convert('L'); a=np.asarray(im).astype(np.float32)
    import cv2
    M=cv2.getRotationMatrix2D((a.shape[1]/2,a.shape[0]/2), random.uniform(-5,5), random.uniform(0.9,1.05))
    a=cv2.warpAffine(a,M,(a.shape[1],a.shape[0]),borderValue=255)
    a=a+np.random.normal(0,8,a.shape); a=np.clip(a,0,255)
    return Image.fromarray(a.astype(np.uint8))

res={}
# ---- TEST 1: CLEAN FD glyphs — same(aug) vs different ----
chars=random.sample(list(fd.keys()),150)
g={c:r._embed_crop(fd[c]) for c in chars}
same1=[cos(g[c], emb_img(aug(Image.open(fd[c])))) for c in chars]
diff1=[cos(g[chars[i]], g[chars[(i+1)%len(chars)]]) for i in range(len(chars))]
res['T1_clean']={'same_mean':round(statistics.mean(same1),3),'diff_mean':round(statistics.mean(diff1),3),
  'same_min':round(min(same1),3),'diff_max':round(max(diff1),3)}

# ---- TEST 2: REAL woodblock crops — same-char vs diff-char ----
rows=[x for x in csv.DictReader(open(f'{D}/labels.csv',encoding='utf-8')) if x['image'] and x['tier']=='GOLD']
bychar=defaultdict(list)
def ink(im): a=np.asarray(im.convert('L')); return (a<128).mean()
for x in rows: bychar[x['label']].append(x['image'])
pairs_same=[]; pairs_diff=[]
multich=[c for c,v in bychar.items() if len(v)>=2][:120]
embcache={}
def cemb(img):
    if img not in embcache:
        im=Image.open(Path(D,img)); 
        embcache[img]=emb_img(im) if ink(im)>=0.08 else None
    return embcache[img]
for c in multich:
    v=random.sample(bychar[c],2); e0,e1=cemb(v[0]),cemb(v[1])
    if e0 is not None and e1 is not None: pairs_same.append(cos(e0,e1))
for i in range(len(multich)-1):
    a=random.choice(bychar[multich[i]]); b=random.choice(bychar[multich[i+1]])
    ea,eb=cemb(a),cemb(b)
    if ea is not None and eb is not None: pairs_diff.append(cos(ea,eb))
res['T2_real']={'same_mean':round(statistics.mean(pairs_same),3),'diff_mean':round(statistics.mean(pairs_diff),3),
  'n_same':len(pairs_same),'n_diff':len(pairs_diff)}

# ---- TEST 3: RETRIEVAL top-1 (crop -> FD gallery of 500 chars) ----
gal_chars=random.sample(list(fd.keys()),500)
gal={c:r._embed_crop(fd[c]) for c in gal_chars}
import numpy as np
galM=np.stack([gal[c].cpu().numpy() for c in gal_chars]); 
hit=0; tot=0
qrows=[x for x in rows if x['label'] in gal_chars]; random.shuffle(qrows)
for x in qrows[:200]:
    e=cemb(x['image'])
    if e is None: continue
    sims=galM@e.cpu().numpy()
    pred=gal_chars[int(np.argmax(sims))]; tot+=1; hit+= (pred==x['label'])
res['T3_retrieval']={'top1_acc':round(hit/max(tot,1),3),'n':tot,'gallery':len(gal_chars),'chance':round(1/len(gal_chars),4)}
print(json.dumps(res,ensure_ascii=False,indent=2))
json.dump(res,open('/tmp/dinov2_proof.json','w'),ensure_ascii=False,indent=2)
