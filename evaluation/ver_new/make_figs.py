import csv, random, json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch.nn.functional as F
import sys; sys.path.insert(0,'.')
from core.ranking.dinov2_ranker import DINOv2Ranker
random.seed(0)
D='evaluation/ver_new/dataset_out'; FIG='evaluation/ver_new/figures'
fd={}
for p in Path('gannhanocr-fd').rglob('U+*.png'):
    try: fd[chr(int(p.stem.replace('U+',''),16))]=str(p)
    except: pass
r=DINOv2Ranker(font_path='font_diffusion/fonts/NomNaTong-Regular.ttf',embedding_cache_dir=f'{D}/emb_cache')
def E(im): return r._embed(im.convert('RGB')).cpu().numpy()
def cos(a,b): return max(0.0,(float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)))+1)/2)
def ink(im): a=np.asarray(im.convert('L')); return (a<128).mean()
rows=[x for x in csv.DictReader(open(f'{D}/labels.csv',encoding='utf-8')) if x['image'] and x['tier']=='GOLD']
bychar=defaultdict(list)
for x in rows: bychar[x['label']].append(x['image'])
ecache={}
def cropE(img):
    if img not in ecache:
        im=Image.open(Path(D,img)); ecache[img]=E(im) if ink(im)>=0.08 else None
    return ecache[img]

# ---- arrays for histograms ----
multich=[c for c,v in bychar.items() if len(v)>=2 and c in fd]
random.shuffle(multich); multich=multich[:200]
same2=[]; diff2=[]
for c in multich:
    v=random.sample(bychar[c],2); e0,e1=cropE(v[0]),cropE(v[1])
    if e0 is not None and e1 is not None: same2.append(cos(e0,e1))
for i in range(len(multich)-1):
    ea=cropE(random.choice(bychar[multich[i]])); eb=cropE(random.choice(bychar[multich[i+1]]))
    if ea is not None and eb is not None: diff2.append(cos(ea,eb))

# ---------- FIG 1: histogram overlap (real crops) ----------
def draw_hist(same,diff,path,title):
    W,H=820,460; L,Rr,T,B=70,30,60,60; pw,ph=W-L-Rr,H-T-B
    cv=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(cv)
    lo,hi=0.5,1.0; nb=40; edges=np.linspace(lo,hi,nb+1)
    hs,_=np.histogram(same,bins=edges); hd,_=np.histogram(diff,bins=edges)
    mx=max(hs.max(),hd.max(),1)
    def X(v): return L+int((v-lo)/(hi-lo)*pw)
    def Y(c): return T+ph-int(c/mx*ph)
    # axes
    d.rectangle([L,T,L+pw,T+ph],outline='black')
    for gv in np.arange(0.5,1.001,0.1):
        x=X(gv); d.line([x,T+ph,x,T+ph+5],fill='black'); d.text((x-12,T+ph+8),f'{gv:.1f}',fill='black')
    # threshold line 0.75
    xt=X(0.75); d.line([xt,T,xt,T+ph],fill=(150,150,150)); d.text((xt-30,T-16),'ngưỡng 0.75',fill=(120,120,120))
    # bars: diff (red) then same (blue, outline) overlaid
    bw=pw/nb
    for i in range(nb):
        x0=L+int(i*bw); x1=L+int((i+1)*bw)
        d.rectangle([x0,Y(hd[i]),x1,T+ph],fill=(255,180,180),outline=None)
    for i in range(nb):
        x0=L+int(i*bw); x1=L+int((i+1)*bw)
        d.rectangle([x0,Y(hs[i]),x1,T+ph],outline=(0,0,200))
    d.text((L,12),title,fill='black')
    d.rectangle([L+pw-180,T+6,L+pw-165,T+18],fill=(255,180,180)); d.text((L+pw-160,T+6),f'khác chữ (n={len(diff)})',fill='black')
    d.rectangle([L+pw-180,T+26,L+pw-165,T+38],outline=(0,0,200)); d.text((L+pw-160,T+26),f'cùng chữ (n={len(same)})',fill=(0,0,200))
    d.text((L,T+ph+30),'cosine (crop ↔ crop)   —   2 phân bố CHỒNG nhau ⇒ không tách được',fill='black')
    cv.save(path)
draw_hist(same2,diff2,f'{FIG}/fig1_hist_real_crops.png',
          f'Hinh 1. Phan bo cosine DINOv2 tren crop THAT: cung chu vs khac chu (mean {np.mean(same2):.3f} vs {np.mean(diff2):.3f})')

# ---------- FIG 2: cosine heatmap of 14 glyphs ----------
gl=random.sample([c for c in fd],14)
ge=[r._embed_crop(fd[c]).cpu().numpy() for c in gl]
N=len(gl); cell=52; pad=58
W=pad+N*cell+20; H=pad+N*cell+20
cv=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(cv)
def gimg(c,s=46):
    im=Image.open(fd[c]).convert('RGB'); im.thumbnail((s,s)); cc=Image.new('RGB',(s,s),'white'); cc.paste(im,((s-im.width)//2,(s-im.height)//2)); return cc
for j,c in enumerate(gl): cv.paste(gimg(c,46),(pad+j*cell+3,4))
for i,c in enumerate(gl): cv.paste(gimg(c,46),(4,pad+i*cell+3))
for i in range(N):
    for j in range(N):
        v=cos(ge[i],ge[j])
        t=max(0.0,min(1.0,(v-0.75)/0.25))  # 0.75->0, 1.0->1
        col=(int(255-t*255),int(255-t*120),int(255-t*120))  # white->red
        x0=pad+j*cell; y0=pad+i*cell
        d.rectangle([x0,y0,x0+cell-2,y0+cell-2],fill=col,outline=(220,220,220))
        d.text((x0+6,y0+18),f'{v:.2f}',fill=(0,0,0) if t<0.6 else (255,255,255))
d.text((6,H-16),'Heatmap cosine giua 14 glyph FD: o ngoai-duong-cheo (khac chu) gan nhu DO dam = duong cheo => khong phan biet',fill='black')
cv.save(f'{FIG}/fig2_cosine_heatmap.png')

# ---------- FIG 3: retrieval failure montage ----------
gal_chars=random.sample(list(fd.keys()),800)
qchars=[c for c in multich if len(bychar[c])>0][:5]
for c in qchars:
    if c not in gal_chars: gal_chars.append(c)
galE=np.stack([r._embed_crop(fd[c]).cpu().numpy() for c in gal_chars])
galN=galE/np.linalg.norm(galE,axis=1,keepdims=True)
sz=92; K=5; rowH=sz+24; W=sz*(K+2)+90; H=rowH*len(qchars)+40
cv=Image.new('RGB',(W,H),'white'); d=ImageDraw.Draw(cv)
d.text((6,6),'Hinh 3. Voi moi CROP (trai), 5 glyph GAN NHAT theo DINOv2. Glyph DUNG (xanh) khong nam trong top.',fill='black')
def crp(img,s=sz,boost=True):
    im=Image.open(Path(D,img)).convert('L')
    if boost: im=ImageOps.autocontrast(im,1)
    im=im.convert('RGB'); im.thumbnail((s,s)); cc=Image.new('RGB',(s,s),'white'); cc.paste(im,((s-im.width)//2,(s-im.height)//2)); return cc
for ri,c in enumerate(qchars):
    img=random.choice(bychar[c]); e=cropE(img)
    if e is None: continue
    y=30+ri*rowH
    cv.paste(crp(img),(10,y)); d.rectangle([10,y,10+sz,y+sz],outline=(200,0,0),width=3); d.text((10,y+sz+2),f'crop = {c}',fill=(200,0,0))
    en=e/np.linalg.norm(e); sims=galN@en; order=np.argsort(-sims)[:K]
    correct_rank=int(np.where(np.argsort(-sims)==gal_chars.index(c))[0][0])+1 if c in gal_chars else -1
    for k,idx in enumerate(order):
        gc=gal_chars[idx]; x=10+sz+20+k*sz
        cv.paste(gimg(gc,sz),(x,y))
        ok=(gc==c)
        d.rectangle([x,y,x+sz,y+sz],outline=(0,170,0) if ok else (120,120,120),width=3 if ok else 1)
        d.text((x,y+sz+2),f'#{k+1} {sims[idx]:.2f}',fill='black')
    # show correct glyph at far right
    x=10+sz+20+K*sz+20
    cv.paste(gimg(c,sz),(x,y)); d.rectangle([x,y,x+sz,y+sz],outline=(0,170,0),width=3)
    d.text((x,y+sz+2),f'DUNG: hang #{correct_rank}',fill=(0,120,0))
cv.save(f'{FIG}/fig3_retrieval_fail.png')
print('figures saved to',FIG)
print('same2 mean',round(np.mean(same2),3),'diff2 mean',round(np.mean(diff2),3))
