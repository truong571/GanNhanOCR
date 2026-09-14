import time, numpy as np, json, sys
from PIL import Image
from paddleocr import PPStructureV3
t=time.time()
pp = PPStructureV3(device="cpu", use_doc_orientation_classify=False, use_doc_unwarping=False, use_textline_orientation=False,
                   use_table_recognition=False, use_seal_recognition=False, use_formula_recognition=False, use_chart_recognition=False)
print("init", time.time()-t)
img = np.array(Image.open("/Users/truongmdn/TruongMDN/ThS/DoAn/GanNhanOCR/prepared/SachThanhTruyen11/pages/page_0010.png").convert("RGB"))[:,:,::-1].copy()
t=time.time(); res=list(pp.predict(img)); print("sec", time.time()-t)
r=res[0]
print(r.keys())
ld=r["layout_det_res"]
for b in ld["boxes"]: print(b["label"], round(b["score"],2), [int(v) for v in b["coordinate"]])
for blk in r["parsing_res_list"][:20]:
    print(getattr(blk,'label',None) or blk.get('block_label'), str(getattr(blk,'content',None) or blk.get('block_content'))[:60])
