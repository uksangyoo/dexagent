"""Goal -> executable graph for pickup experiments, using a restricted hand API.

Policies are authored in this conversation. No separate model invocation or
agent-population ranking is claimed. Success remains an independent oracle.
"""
import hashlib,importlib.util,json
from pathlib import Path
from .geometry import ART,ROOT

INTERFACES={
 'cartesian':dict(command='fingertip_hold',description='Hold fingertip locations. Contact-normal pressure uses a shared inner servo; no fingertip angular command. Pivot uses a fixed release pressure.'),
 'object':dict(command='rigid_grasp_twist',description='Map a desired tool twist into fingertip translation and angular velocity. The rigid-grasp target includes all three angular coordinates.'),
 'hom':dict(command='relative_contact',description='Hold tangential contact locations, regulate pressure, constrain contact-normal spin during lift, then release it and brake gravity pivot with pressure feedback.')}

def build(task,interface):
 if task not in ['lift_hold','controlled_pivot','pickup_pivot_drive']:raise ValueError('Unknown task')
 if interface not in INTERFACES:raise ValueError(interface)
 destination=ART/'policies'/f'{task}-{interface}';destination.mkdir(parents=True,exist_ok=True)
 goal='Pick up the horizontal screwdriver, lift it 140 mm and hold its original orientation.'if task=='lift_hold'else'Pick up the screwdriver, allow loaded in-hand slipping to pivot it upright, then hold it with index-cap support.'
 if task=='pickup_pivot_drive':goal='Pick up the screwdriver, pivot it through loaded in-hand slip, engage the slotted screw, and drive it to its seat.'
 nodes={
  'pinch':{'action':'pinch','edges':[('loaded_s','gt',.15,'lift')],'otherwise':'pinch'},
  'lift':{'action':'lift','edges':[('lift_ready','eq',True,'hold_horizontal'if task=='lift_hold'else'release')],'otherwise':'lift'},
  'hold_horizontal':{'action':'hold_horizontal','edges':[('lift_accepted','eq',True,'succeeded')],'otherwise':'hold_horizontal'},
  'release':{'action':'release','edges':[('release_ready','eq',True,'pivot')],'otherwise':'release'},
  'pivot':{'action':'pivot','edges':[('tilt_deg','lt',35,'cap_support')],'otherwise':'pivot'},
  'cap_support':{'action':'cap_support','edges':[('pivot_accepted','eq',True,'succeeded')],'otherwise':'cap_support'},
  'succeeded':{'terminal':'succeeded'},'failed':{'terminal':'failed'}}
 if task=='pickup_pivot_drive':
  nodes['cap_support']['edges']=[('pivot_accepted','eq',True,'assemble')]
  nodes['assemble']=dict(action='assemble',edges=[('assembly_accepted','eq',True,'succeeded')],otherwise='assemble',subgraph=['transport','approach','drive','drive_hold'],guards='measured tip alignment, bit contact, screw insertion and sustained seating')
 graph=dict(format='dexagent.pickup-policy/1',task=task,goal=goal,abstraction=interface,interface=INTERFACES[interface],start='pinch',nodes=nodes,graph_hz=20,physics_edit_access=False)
 code='"""Generated goal policy. Observation inputs only; motor control is behind the hand API."""\n\n'
 for name,node in nodes.items():
  code+=f'def {name}(o):\n'
  if 'terminal'in node:code+=f'    return {name!r}\n\n';continue
  code+='    if o["failure"]: return "failed"\n'
  for key,op,value,target in node['edges']:
   operator={'gt':'>','lt':'<','eq':'=='}[op];code+=f'    if o[{key!r}] {operator} {value!r}: return {target!r}\n'
  code+=f'    return {node["otherwise"]!r}\n\n'
 code+='NODES = {'+', '.join(f'{n!r}: {n}'for n in nodes)+'}\n'
 (destination/'graph.json').write_text(json.dumps(graph,indent=2));(destination/'policy.py').write_text(code)
 digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
 manifest=dict(goal=goal,task=task,abstraction=interface,author='Codex in this conversation',runtime_model_invocation=False,video_goal_source='reference/screwdriver-five_hands-compressed.mp4',video_sha256=digest(ROOT/'reference/screwdriver-five_hands-compressed.mp4'),code_sha256=digest(destination/'policy.py'),graph_sha256=digest(destination/'graph.json'),initial_condition='collision-free horizontal tool on two supports; not flat-table pickup',full_sequence_supported=task=='pickup_pivot_drive',full_sequence_validated=False)
 (destination/'manifest.json').write_text(json.dumps(manifest,indent=2));return destination

class Policy:
 def __init__(self,path):
  self.path=Path(path);self.graph=json.loads((self.path/'graph.json').read_text());self.manifest=json.loads((self.path/'manifest.json').read_text())
  for node in self.graph['nodes'].values():
   if 'command'in node and self.graph['task']=='pickup_pivot_drive':
    from .workflow import validate_command
    validate_command(node['command'],self.graph['abstraction'])
  for name,key in [('policy.py','code_sha256'),('graph.json','graph_sha256')]:
   if hashlib.sha256((self.path/name).read_bytes()).hexdigest()!=self.manifest[key]:raise ValueError('Modified policy artifact')
  spec=importlib.util.spec_from_file_location('pickup_goal_policy',self.path/'policy.py');self.module=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.module)
  self.node=self.graph['start'];self.trace=[];self.ticks=[]
 def step(self,observation):
  target=self.module.NODES[self.node](observation)
  if target not in self.graph['nodes']:raise ValueError('Invalid graph edge')
  if target!=self.node:self.trace.append(dict(t=observation['t'],from_node=self.node,to=target))
  self.node=target;decision=dict(node=target,**self.graph['nodes'][target]);self.ticks.append(dict(observation=observation,decision=decision));return decision

if __name__=='__main__':
 for task in ['lift_hold','controlled_pivot']:
  for interface in INTERFACES:print(build(task,interface))
