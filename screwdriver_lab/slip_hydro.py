"""Solved hydroelastic patch observations for index-cap / thumb-middle roles."""
import numpy as np
from scipy.spatial.transform import Rotation
from screwdriver_lab.hydro_backend import HydroBackend
class SlipHydroBackend(HydroBackend):
 def measure(self):
  super().measure();s=self.s;s.handleloads=[0.,0.];s.slip=[0.,0.];s.signed_slip=[0.,0.];s.index_cap_load=0.;s.rigid_tool_load=0.;s.pad_points=np.zeros((2,3));s.pad_normals=np.zeros((2,3));s.pad_weights=np.zeros(2)
  c=self.solved;n=int(c.rigid_contact_count.numpy()[0]);a=c.rigid_contact_shape0.numpy()[:n];b=c.rigid_contact_shape1.numpy()[:n];p0=c.rigid_contact_point0.numpy()[:n];p1=c.rigid_contact_point1.numpy()[:n];norm=c.rigid_contact_normal.numpy()[:n];forces=c.force.numpy()[:n,:3];R=Rotation.from_quat(self.sample_q[:,3:]).as_matrix();pos=self.sample_q[:,:3];axis=s.d.xmat[s.tool].reshape(3,3)[:,2]
  for k,(a0,b0)in enumerate(zip(a,b)):
   if min(a0,b0)<0:continue
   bodies=[int(self.sb[a0]),int(self.sb[b0])];gn=[self.shape_names[a0],self.shape_names[b0]]
   if not any(x in gn for x in ['handle','cap']):continue
   target='handle'if'handle'in gn else'cap';obj=gn.index(target);other=1-obj;body=bodies[other];name=gn[other];load=abs(float(np.dot(forces[k],norm[k])))
   if body<0:continue
   if self.body_names[body].startswith('link_')and not name.startswith('finger_tip_'):s.rigid_tool_load+=load
   if target=='cap'and name=='finger_tip_2':s.index_cap_load+=load
   if target!='handle'or name not in ['finger_tip_1','finger_tip_3']:continue
   slot=0 if name=='finger_tip_1'else 1;s.handleloads[slot]+=load
   pa=pos[bodies[0]]+R[bodies[0]]@p0[k];pb=pos[bodies[1]]+R[bodies[1]]@p1[k];point=(pa+pb)/2;out=norm[k]*(1 if obj==0 else -1)
   # Orient normals geometrically; backend sign convention cannot invert control.
   radial=point-s.d.xpos[s.tool];radial-=axis*np.dot(radial,axis)
   if np.dot(out,radial)<0:out=-out
   weight=max(load,1e-8);s.pad_points[slot]+=weight*point;s.pad_normals[slot]+=weight*out;s.pad_weights[slot]+=weight
   vel=[self.sample_v[b,:3]+np.cross(self.sample_v[b,3:],point-pos[b]-R[b]@self.com[b])for b in bodies];v=vel[other]-vel[obj];tangent=-np.cross(axis,radial/max(1e-9,np.linalg.norm(radial)));vt=v-out*np.dot(v,out)
   s.slip[slot]+=load*np.linalg.norm(vt)*1000;s.signed_slip[slot]+=load*np.dot(v,tangent)*1000
  for i in range(2):
   if s.pad_weights[i]>0:s.pad_points[i]/=s.pad_weights[i];s.pad_normals[i]/=max(1e-9,np.linalg.norm(s.pad_normals[i]))
   s.slip[i]/=max(1e-9,s.handleloads[i]);s.signed_slip[i]/=max(1e-9,s.handleloads[i])
