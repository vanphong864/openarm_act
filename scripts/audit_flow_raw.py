"""Audit small state arrays and sampled depth frames without changing raw data."""
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import h5py
import numpy as np
from openarm_data.episode import EpisodeReader

root = Path(__file__).resolve().parents[1]
rows = []
for p in sorted((root/'data/raw/data_openarm_8_10').glob('*.hdf5')):
    row = {'episode': p.name}
    try:
        with EpisodeReader(p) as r:
            q = r.qpos()
            labels = r.labels('commands')
            arms = [i for i in range(16) if i not in (7,15)]
            dq = np.abs(np.diff(q[:, arms], axis=0))
            row.update(frames=len(q), source=labels.source, notes=labels.notes,
                       nonfinite_state=int((~np.isfinite(q)).sum()),
                       max_arm_step_rad=float(np.nanmax(dq)),
                       large_arm_steps=int((dq > .15).any(axis=1).sum()),
                       gripper_min_m=q[:,[7,15]].min(axis=0).tolist(),
                       gripper_max_m=q[:,[7,15]].max(axis=0).tolist())
            sampled=[]
            for i in np.linspace(0,len(q)-1,5,dtype=int):
                d=r.depth(i)
                sampled.append(float(((d>200)&(d<1200)).mean()))
            row['sampled_depth_valid_fraction']=sampled
            if 'timestamp_ns' in r.file:
                t=np.asarray(r.file['timestamp_ns'][:]).reshape(-1)
                dt=np.diff(t.astype('float64'))/1e9
                row['timestamp_nonpositive']=int((dt<=0).sum())
                row['timestamp_dt_p50_s']=float(np.median(dt))
                row['timestamp_gaps_over_100ms']=int((dt>.1).sum())
    except Exception as exc:
        row['error']=str(exc)
    rows.append(row)
out=root/'outputs/raw_audit.json'
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(rows,indent=2))
print('episodes',len(rows),'errors',sum('error' in r for r in rows),flush=True)
print('sources', {s:sum(r.get('source')==s for r in rows) for s in ('commands','next_qpos')})
for r in rows:
    if 'error' in r or r.get('large_arm_steps',0) or r.get('timestamp_nonpositive',0):
        print(json.dumps(r))
valid=[x for r in rows for x in r.get('sampled_depth_valid_fraction',[])]
print('depth valid min/median/max',np.percentile(valid,[0,50,100]))
print('frames',sum(r.get('frames',0) for r in rows),'report',out)
