"""Per-physics-step guards retained inside CUDA graph batching."""
import warp as wp
@wp.kernel
def clear_loads(loads:wp.array[float]):
 loads[wp.tid()]=0.
@wp.kernel
def contacts_watch(count:wp.array[int],a:wp.array[int],b:wp.array[int],p0:wp.array[wp.vec3],p1:wp.array[wp.vec3],norm:wp.array[wp.vec3],force:wp.array[wp.spatial_vector],shape_body:wp.array[int],body_q:wp.array[wp.transform],finger:wp.array[int],target:wp.array[int],loads:wp.array[float],guard:wp.array[float]):
 i=wp.tid()
 if i>=count[0]:return
 sa=a[i];sb=b[i]
 if sa<0 or sb<0:return
 ba=shape_body[sa];bb=shape_body[sb];pa=p0[i];pb=p1[i]
 if ba>=0:pa=wp.transform_point(body_q[ba],pa)
 if bb>=0:pb=wp.transform_point(body_q[bb],pb)
 wp.atomic_max(guard,1,-wp.dot(pb-pa,norm[i]))
 f=force[i];fn=wp.abs(wp.dot(wp.vec3(f[0],f[1],f[2]),norm[i]))
 if target[sa]>0 and finger[sb]>=0:wp.atomic_add(loads,finger[sb],fn)
 if target[sb]>0 and finger[sa]>=0:wp.atomic_add(loads,finger[sa],fn)
@wp.kernel
def record(loads:wp.array[float],count:wp.array[int],guard:wp.array[float]):
 i=wp.tid();wp.atomic_max(guard,0,loads[i]);wp.atomic_max(guard,2,float(count[0]))
@wp.kernel
def capacity_watch(count:wp.array[int],capacity:int,guard:wp.array[float]):
 if count[0]>capacity:guard[3]=1.
 wp.atomic_max(guard,4,float(count[0])/float(capacity))
