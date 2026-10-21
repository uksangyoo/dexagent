"""Task measurements independent of the goal graph's terminal declaration."""
import numpy as np
from scipy.spatial.transform import Rotation

def orientation(cube_R,palm_R,initial_cube_R,initial_palm_R,target_world_R):
 relative=palm_R.T@cube_R
 target_relative=initial_palm_R.T@target_world_R
 initial_relative=initial_palm_R.T@initial_cube_R
 angle=lambda R:float(np.degrees(Rotation.from_matrix(R).magnitude()))
 return dict(error_deg=angle(target_relative@relative.T),inhand_turn_deg=angle(relative@initial_relative.T),palm_turn_deg=angle(palm_R@initial_palm_R.T))

def valid_hold(metrics,lift_mm,omega_deg_s,side_loads,unexpected_N):
 return bool(metrics['error_deg']<=5 and metrics['palm_turn_deg']<2 and lift_mm>=60 and omega_deg_s<5 and min(side_loads)>.05 and unexpected_N<.1)
