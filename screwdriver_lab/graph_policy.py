"""Small GaP-style graph executor; independent of the GaP benchmark harness.

A policy artifact owns nodes, typed state references, conditional exits and
backward edges. Servo implementations are ordinary registered Python skills.
The runtime records every graph transition with the observations that caused it.
"""
import json
from pathlib import Path
PHASES=['approach','grasp','lift','reorient','align','press','shift_grip','reposition','turn','seat']
PATH=Path(__file__).with_name('graph')/'workflow.json'

def author():
 descriptions={
 'approach':'Move the UR5e to the horizontal screwdriver with the Tesollo hand open.',
 'grasp':'Acquire opposed handle contacts and regulate normal force.',
 'lift':'Lift clear of both cradle supports; verify retained tool support.',
 'reorient':'Bring the tool upright with arm pose and grasp feedback.',
 'align':'Align the actual bit with the screw slot before descending.',
 'press':'Establish measured bit-slot contact under a bounded axial load.',
 'shift_grip':'Maintain the pinch and acquire the swivel cap with the thumb.',
 'reposition':'Move the arm and slide the side contacts while the thumb maintains cap pressure.',
 'turn':'Drive, slide or release, return, and reacquire contact patches.',
 'seat':'Stop angular motion and verify depth, axial load, and head seating.'}
 w={'version':3,'meta':{'name':'ur5e_tesollo_screw_in','description':'Agent-authored graph with interchangeable dexterous-hand servo skills. No GaP harness or runtime LLM required.','structure_source':'graph-as-policy/examples/build_a_graph and grocery_packing/packing_graph','source_commit':'c8b5515df80d3029ca5dc7f7f1ec1c327b70bbea'},'nodes':{},'edges':[['START','approach']],'conditional_edges':{},'subgraphs':{}}
 for i,name in enumerate(PHASES):
  w['nodes'][name]={'type':'subgraph','ref':name+'_sg'}
  w['conditional_edges'][name]={'router_field':'exit','mapping':{'complete':PHASES[i+1]if i+1<len(PHASES)else'done','failed':'abort'}}
  w['subgraphs'][name+'_sg']={'skill':'dexterous-'+name,'description':descriptions[name],'inputs':{'observation':{'type':'ArmHandObservation'},'abstraction':{'type':'ControllerInterface'}},'outputs':{'state':{'$ref':'tick.state'}},'nodes':{'observe':{'type':'tool','tool':'robot.observe'},'tick':{'type':'tool','tool':'controller.'+name,'inputs':{'observation':{'$ref':'observe.observation'},'abstraction':{'$ref':'in.abstraction'}}},'complete':{'type':'exit'},'failed':{'type':'exit'}},'edges':[['START','observe'],['observe','tick'],['complete','END'],['failed','END']],'conditional_edges':{'tick':{'router_field':'exit','mapping':{'running':'observe','complete':'complete','failed':'failed'}}}}
 w['conditional_edges']['turn']['mapping']['reacquire']='shift_grip'
 w['nodes']['done']={'type':'end','status':'success'};w['nodes']['abort']={'type':'end','status':'failure'}
 PATH.write_text(json.dumps(w,indent=2));return w

class GraphPolicy:
 def __init__(self):
  self.spec=json.loads(PATH.read_text());self.current=self.spec['edges'][0][1];self.transitions=[]
 def route(self,event,t,observation,reason):
  old=self.current;self.current=self.spec['conditional_edges'][old]['mapping'][event]
  keys=['depth_mm','finger_forces','axial_force_N','bit_lateral_mm','bit_height_mm','tilt_deg','engaged','arm_pose_error_mm','slip_mm_s']
  self.transitions.append({'t':t,'from':old,'exit':event,'to':self.current,'reason':reason,'evidence':{k:observation[k]for k in keys if k in observation}})
  return self.current
if __name__=='__main__':author()
