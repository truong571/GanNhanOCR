import json, sys
from pathlib import Path
D = Path(__file__).resolve().parents[3] / "measure_out/_tn6/x02"
for v in sys.argv[1:]:
    d = json.load(open(D / f"{v}.json"))
    for ps, r in d["res"].items():
        print(f"{v:10s} {ps:7s}", " | ".join(f"{b[:4]} new {x['new']:.4f} pitch {x['pitch']:.4f} gold {x['gold_new']:.4f} v5all {x['all_items']['slot_v5']:.4f} skip {x['kinds'].get('skip',0)}" for b, x in r.items()))
