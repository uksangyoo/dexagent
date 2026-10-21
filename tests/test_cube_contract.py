import unittest
import tempfile,json,hashlib
from pathlib import Path
from unittest.mock import patch
import numpy as np
from scipy.spatial.transform import Rotation
from tasks.cube.metrics import orientation,valid_hold
from tasks.cube.policy import build,Policy
from tasks.cube.workflow import prepare,compile_policy,digest

class CubeContract(unittest.TestCase):
 def test_agent_compiler_locks_goal_and_rejects_physics_arguments(self):
  with tempfile.TemporaryDirectory()as directory,patch('tasks.cube.policy.OUT',Path(directory)):
   root=Path(directory);request=prepare('Pick up and rotate the cube',root/'request')
   seed=build('hom',30,.04,'y');g=json.loads((seed/'graph.json').read_text());candidate={k:g[k]for k in ['abstraction','start','nodes']};candidate['request_sha256']=digest(request)
   out=compile_policy(candidate,request,root/'compiled','unit-test fixture');p=Policy(out)
   self.assertEqual(p.graph['contract'],json.loads(request.read_text())['contract'])
   candidate['nodes']['rotate']['command']['arguments']['object_torque']=1
   with self.assertRaisesRegex(ValueError,'physics edits forbidden'):compile_policy(candidate,request,root/'bad','unit-test fixture')
 def test_graph_emits_bounded_relative_contact_command(self):
  with tempfile.TemporaryDirectory()as directory,patch('tasks.cube.policy.OUT',Path(directory)):
   path=build('hom',30,.04,'y',dict(grip=.6,speed=.2,pusher_offset=.015,spin_gain=.06,contact_mode='spin'),'trial')
   p=Policy(path);command=p.step(dict(t=0,failure=False,grasp_ready=False))['command']
   self.assertEqual(command['operation'],'relative_contact');self.assertEqual(command['arguments']['pusher_offset'],.015)
   graph=json.loads((path/'graph.json').read_text());graph['nodes']['rotate']['command']['arguments']['grip']=100
   (path/'graph.json').write_text(json.dumps(graph));manifest=json.loads((path/'manifest.json').read_text());manifest['graph_sha256']=hashlib.sha256((path/'graph.json').read_bytes()).hexdigest();(path/'manifest.json').write_text(json.dumps(manifest))
   with self.assertRaisesRegex(ValueError,'bounded hand command'):Policy(path)
 def test_wrist_rotation_is_not_inhand_rotation(self):
  I=np.eye(3);R=Rotation.from_euler('z',90,degrees=True).as_matrix()
  a=orientation(R,R,I,I,R)
  self.assertAlmostEqual(a['inhand_turn_deg'],0);self.assertAlmostEqual(a['error_deg'],90)
  self.assertFalse(valid_hold(a,80,0,[.6,.6],0))
 def test_wrong_axis_quarter_turn_fails(self):
  I=np.eye(3);goal=Rotation.from_euler('z',90,degrees=True).as_matrix();wrong=Rotation.from_euler('x',90,degrees=True).as_matrix()
  a=orientation(wrong,I,I,I,goal)
  self.assertAlmostEqual(a['inhand_turn_deg'],90);self.assertGreater(a['error_deg'],90)
  self.assertFalse(valid_hold(a,80,0,[.6,.6],0))
 def test_goal_pose_requires_loaded_hold(self):
  I=np.eye(3);R=Rotation.from_euler('z',90,degrees=True).as_matrix();a=orientation(R,I,I,I,R)
  self.assertTrue(valid_hold(a,80,0,[.6,.6],0))
  for lift,speed,loads,force in [(0,0,[.6,.6],0),(80,30,[.6,.6],0),(80,0,[0,.6],0),(80,0,[.6,.6],1)]:self.assertFalse(valid_hold(a,lift,speed,loads,force))
if __name__=='__main__':unittest.main()
