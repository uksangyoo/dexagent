"""Robot-only resolved-rate contact admittance; integrated joint commands.

Normal load error produces a radial velocity. Tangential motion and axial
tracking are resolved through the measured fingertip Jacobian. Joint targets
are integrated and tracked by bounded position servos; no object force exists.
"""
import numpy as np,mujoco

def command(c):
    s=c.s
    if not hasattr(c,'rate_q'):
        c.rate_q=s.d.qpos[s.qi].copy();s.set_hand_impedance(8.,.18)
        c.rate_load=np.asarray(s.handleloads).copy()
    c.rate_load+=(1-np.exp(-.001/.02))*(np.asarray(s.handleloads)-c.rate_load)
    if hasattr(s,'task_effort'):s.task_effort={}
    axis=s.d.xmat[s.tool].reshape(3,3)[:,2];center=s.d.xpos[s.tool]
    reference=c.drive_center-np.array([0,0,s.d.qpos[s.depthq]-c.drive_depth])
    for slot,i in enumerate([0,2]):
        b=s.bids[i];R=s.d.xmat[b].reshape(3,3);sl=slice(4*i,4*i+4);cols=s.vi[sl]
        if not hasattr(c,'rate_local'):c.rate_local=[v.copy()for v in c.local]
        if s.handleloads[slot]>.001:c.rate_local[slot]+=.1*(R.T@(s.pad_points[slot]-s.d.xpos[b])-c.rate_local[slot])
        point=s.d.xpos[b]+R@c.rate_local[slot];radial=point-center;radial-=axis*np.dot(radial,axis);rn=radial/max(1e-9,np.linalg.norm(radial));tangent=-np.cross(axis,rn)
        angle=c.angle[slot]if np.ndim(c.angle)else c.angle;phi=c.phi[slot]-np.radians(angle)
        desired=reference+.012*np.array([np.cos(phi),np.sin(phi),0.])+c.height[slot]*np.array([0.,0.,1.])
        speed=np.radians(c.command_speed[slot])*.012
        velocity=tangent*np.clip(speed+4*np.dot(desired-point,tangent),-.006,.006)
        velocity+=axis*np.clip(4*np.dot(desired-point,axis),-.003,.003)
        velocity-=rn*np.clip(.002*(c.force_targets[slot]-c.rate_load[slot]),-.003,.003)
        J=np.zeros((3,s.m.nv));Jr=J.copy();mujoco.mj_jac(s.m,s.d,J,Jr,point,b);Ji=J[:,cols]
        pinv=Ji.T@np.linalg.inv(Ji@Ji.T+1e-7*np.eye(3));null=np.eye(4)-pinv@Ji
        external=J@s.d.qvel-Ji@s.d.qvel[cols]
        normal=R@c.normal_local[slot]
        qd=pinv@(velocity-external)+null@(Jr[:,cols].T@(2*np.cross(normal,-rn))+.1*(c.posture_ref[sl]-s.d.qpos[s.qi[sl]]))
        c.rate_q[sl]+=np.clip(qd,-1.,1.)*.001
        c.rate_q[sl]=np.clip(c.rate_q[sl],s.d.qpos[s.qi[sl]]-.025,s.d.qpos[s.qi[sl]]+.025)
        s.d.ctrl[sl]=np.clip(c.rate_q[sl],s.m.jnt_range[s.jids[sl],0]+1e-5,s.m.jnt_range[s.jids[sl],1]-1e-5)
