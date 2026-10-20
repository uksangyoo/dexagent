"""One-command, contact-driven pickup through screw seating on CPU MuJoCo.

The exact-state handoff is an experiment boundary, not a state reset. It
preserves the full integration state and the held robot motor command.
Newton GPU validation reaches engagement; full screw-in remains unverified.
"""
import os
os.environ['CUDA_VISIBLE_DEVICES']='1'
import argparse,json,subprocess,sys
from screwdriver_lab.physical_task import OUT
from screwdriver_lab.drive_checkpoint import run
from screwdriver_lab.physical_evidence import join

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--name',required=True)
    p.add_argument('--interface',choices=['pose','velocity','wrench'],default='wrench')
    p.add_argument('--strategy',choices=['paired','alternating'],default='alternating')
    p.add_argument('--preset',choices=['orientation','axis','recovery'],default='orientation',help='Alternating controller refinement; orientation is the verified nominal configuration.')
    p.add_argument('--initial-y-mm',type=float,default=0)
    p.add_argument('--duration',type=float,default=200)
    a=p.parse_args();prefix=a.name+'-pickup'
    # Prevent accidental overwrite of a completed experiment's evidence.
    if (OUT/(a.name+'.json')).exists()or(OUT/(prefix+'.json')).exists():
        raise SystemExit('Choose a new experiment name; existing evidence is retained.')
    subprocess.run([sys.executable,'-m','screwdriver_lab.physical_task','--name',prefix,
        '--diagnostic-cpu','--checkpoint-only','--timeout','40','--initial-y-mm',str(a.initial_y_mm)],check=True)
    record=json.loads((OUT/(prefix+'.json')).read_text())
    if record['outcome']!='checkpoint_ready':
        raise SystemExit('Physical pickup/engagement failed: '+record['outcome'])
    options=dict(interface=a.interface,duration=a.duration,cycle=a.strategy,
        cap_compensation=False,drive_normal=.6,return_normal=.03,cap_load=2.)
    if a.strategy=='alternating':
        options.update(drive_normal=.9,return_normal=.3,cap_load=1.5,
            cap_lateral=.75,cap_stiffness=1000,cap_damping=12,orientation=.08,
            effort_seating=True,tangent_k=600,tangent_d=8,
            side_comp=.5,force_p=0,force_i=.5)
        if a.preset in ['axis','recovery']:options.update(cap_damping=3,orientation=.015)
        if a.preset=='recovery':options['axis_recovery']=True
    run(prefix,a.name,**options)
    full,_,_=join(a.name)
    print(json.dumps(dict(id=full['id'],outcome=full['outcome'],success=full['success'],
        screw_angle_deg=full['final']['screw_angle_deg'],depth_mm=full['final']['depth_mm'],
        loaded_return_fraction=full['metrics']['return_loaded_fraction'],
        continuity=full['continuity_audit']),indent=2))
if __name__=='__main__':main()
