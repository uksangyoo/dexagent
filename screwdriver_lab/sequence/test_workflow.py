import json,tempfile,unittest
from pathlib import Path
from .workflow import prepare,compile_policy,digest
from .policy import Policy

class SequenceContract(unittest.TestCase):
 def test_video_goal_binding_and_physics_edits_rejected(self):
  with tempfile.TemporaryDirectory()as directory:
   root=Path(directory);video=root/'reference.mp4';video.write_bytes(b'test video identity')
   req=prepare('Pick up, pivot through slip, and seat the screw',video,root/'request')
   node=dict(action='assemble',edges=[['assembly_accepted','eq',True,'succeeded']],otherwise='drive',command=dict(operation='relative_contact',arguments=dict(cycle='paired',drive_load=.4,return_load=.04,speed=30)))
   candidate=dict(request_sha256=digest(req),abstraction='hom',start='drive',nodes=dict(drive=node,succeeded=dict(terminal='succeeded'),failed=dict(terminal='failed')))
   output=compile_policy(candidate,req,root/'policy','test fixture');p=Policy(output)
   self.assertEqual(p.manifest['video_sha256'],digest(video))
   self.assertEqual(p.step(dict(t=0,failure=False,assembly_accepted=False))['command'],node['command'])
   self.assertEqual(p.step(dict(t=.05,failure=False,assembly_accepted=True))['terminal'],'succeeded')
   node['command']['arguments']['object_torque']=.1
   with self.assertRaisesRegex(ValueError,'physics edits forbidden'):compile_policy(candidate,req,root/'bad','test fixture')

if __name__=='__main__':unittest.main()
