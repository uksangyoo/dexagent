"""Nominal graph refinement: wait for measured screw motion at stroke end."""
import json,hashlib
from pathlib import Path
import screwdriver_lab.controlled_slip as cs
from screwdriver_lab.slip_compare import BASE,METHODS
from screwdriver_lab.slip_graph import SlipPolicyGraph
class ProgressGraph(SlipPolicyGraph):
 def __init__(self):
  super().__init__();self.schema['meta']['name']='measured_progress_slip_skill';self.schema['meta']['drive_exit']='command stroke complete AND measured screw advance >=12 deg, or a 4 s drive timeout'
 def drive(self,c,x):
  c.angle=min(c.amplitude,c.angle+c.speed*.02)
  moved=c.s.observe()['screw_angle_deg']-c.stroke_start_angle
  if c.angle>=c.amplitude and (moved>=12 or x['t']-c.start>=4):return 'stroke_complete'
class ProgressController(cs.SlipPairedController):
 def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.graph=ProgressGraph();self.stroke_start_angle=0
 def transition(self,phase,o):
  super().transition(phase,o)
  if phase=='drive':self.stroke_start_angle=o['screw_angle_deg']
if __name__=='__main__':
 cs.SlipPairedController=ProgressController;r=cs.run('slip-progress-graph',**{**BASE,**METHODS['wrench'],'speed':36,'amplitude':30});p=Path(__file__);sha=hashlib.sha256(p.read_bytes()).hexdigest();r['source_sha256'][p.name]=sha;(cs.OUT/'source_snapshots'/(sha+'_'+p.name)).write_bytes(p.read_bytes());r['parameters'].update(controller_class='ProgressController',minimum_stroke_progress_deg=12,drive_timeout_s=4);(cs.OUT/(r['id']+'.json')).write_text(json.dumps(r))
