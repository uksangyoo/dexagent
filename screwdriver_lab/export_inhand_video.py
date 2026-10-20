"""Export exact physics samples, with explicit per-segment playback scales."""
import argparse,json,numpy as np
from screwdriver_lab.full_scene import OUT
from screwdriver_lab.export_rtx import export
from screwdriver_lab.audit_inhand import audit
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('--intro-only',action='store_true');p.add_argument('--pivot-only',action='store_true');p.add_argument('--spin-scale',type=float,default=30);p.add_argument('--fps',type=int,default=15);p.add_argument('--output');a=p.parse_args()
r,summary=audit(a.name);q=np.load(OUT/(a.name+'.npz'))['qpos'];ts=np.array([f['t']for f in r['frames']]);end=float(ts[-1]);drive=summary['drive_start_s'] or end
segments=[(0,min(drive+.3,end),2.)]
if a.pivot_only:segments=[(max(0,summary['pivot_start_s']-.2),min(end,summary['pivot_end_s']+.2),1.)]
elif not a.intro_only and segments[0][1]<end:segments.append((segments[0][1],end,a.spin_scale))
ix=[];scales=[];mapping=[]
for lo,hi,scale in segments:
 first=len(ix)
 for t in np.arange(lo,hi,scale/a.fps):ix.append(int(np.argmin(abs(ts-t))));scales.append(scale)
 mapping.append(dict(video_start_s=first/a.fps,simulation_start_s=lo,simulation_end_s=hi,time_scale=scale))
ix.append(int(np.argmin(abs(ts-segments[-1][1]))));scales.append(scales[-1]);r['frames']=[dict(r['frames'][i],display_scale=scale,source_frame_index=i)for i,scale in zip(ix,scales)];r['method']='IN-HAND PIVOT + SPIN';r['video_time_scale']=1;r['presentation_segments']=mapping;r['pose_audit']=summary;r['source_outcome']=r['outcome'];r['source_id']=a.name;r['witness_flag']='Purple flag is a display-only rotation witness parented to the recorded handle; absent from collision dynamics.'
if a.intro_only or a.pivot_only:r['outcome']='excerpt end'
for k in ['commands','trace']:r.pop(k,None)
name=a.output or a.name+('-intro'if a.intro_only else'-pivot'if a.pivot_only else'-overview');export((OUT/(a.name+'.xml')).read_text(),q[ix],OUT/'rtx'/name,r);print(name,len(ix),mapping)
