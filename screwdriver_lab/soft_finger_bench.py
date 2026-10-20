"""Independent torque-threshold check for the declared soft-finger law."""
import json
from pathlib import Path
import numpy as np,mujoco
from screwdriver_lab.soft_finger import torsion_length,description

def run(load,dt=.00025,impratio=1):
    mass=load/9.81;radius=.01;mu=.6;length=torsion_length(load,mu,1e9)
    xml=f'''<mujoco><option timestep="{dt}" gravity="0 0 -9.81" cone="elliptic" impratio="{impratio}"
      integrator="implicitfast" iterations="100"/>
      <default><geom solref=".004 1" solimp=".99 .99 .001"/></default>
      <worldbody><geom type="plane" size="1 1 .01"/>
      <body pos="0 0 .0101"><freejoint/><geom type="sphere" size=".01"
      mass="{mass}" priority="1" condim="4" friction="{mu} {length} 0"/>
      </body></worldbody></mujoco>'''
    m=mujoco.MjModel.from_xml_string(xml);d=mujoco.MjData(m);trace=[]
    for k in range(round(.8/dt)):
        ratio=0 if d.time<.3 or d.time>=.6 else .5 if d.time<.5 else 1.5
        torque=ratio*length*load;d.xfrc_applied[1,5]=torque
        mujoco.mj_step(m,d)
        f=np.zeros(6)
        if d.ncon:mujoco.mj_contactForce(m,d,0,f)
        trace.append([d.time,d.qvel[-1],torque,f[0],f[3],d.qpos[2]])
    a=np.array(trace);sub=a[(a[:,0]>.4)&(a[:,0]<.49)];over=a[(a[:,0]>.55)&(a[:,0]<.59)]
    inertia=.4*mass*radius**2;actual_acceleration=np.polyfit(over[:,0],over[:,1],1)[0]
    expected_acceleration=.5*length*load/inertia
    creep=max(abs(sub[:,1]))*length/(3*np.pi/16*mu)
    return dict(load_N=load,dt=dt,impratio=impratio,torsion_coefficient_m=length,threshold_Nm=length*load,
      subthreshold_patch_edge_creep_um_s=creep*1e6,
      subthreshold_max_angular_speed=float(max(abs(sub[:,1]))),
      superthreshold_angular_acceleration=actual_acceleration,expected_acceleration=expected_acceleration,
      acceleration_relative_error=abs(actual_acceleration/expected_acceleration-1),
      normal_load_relative_error=abs(sub[:,3].mean()/load-1),
      passed=bool(creep<1e-6 and abs(actual_acceleration/expected_acceleration-1)<.05)),a

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--impratio',type=float,default=1);args=p.parse_args()
    out=Path(__file__).parent/'artifacts/physical_task';results=[]
    for load in [.25,1.,2.]:
        for dt in [.00025,.000125]:
            r,a=run(load,dt,args.impratio);results.append(r);np.savez_compressed(out/f'soft-bench-i{args.impratio}-{load}-{dt}.npz',trace=a)
    (out/f'soft-contact-validation-i{args.impratio}.json').write_text(json.dumps(dict(model=description(),cases=results),indent=2))
    print(json.dumps(results,indent=2))
