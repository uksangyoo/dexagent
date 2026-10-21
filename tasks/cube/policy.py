"""Executable, observation-routed cube goal graphs authored by Codex here."""
import json,hashlib
from tasks.cube.model import OUT
from dexagent.planning.policy import Policy as GraphPolicy

LIMITS={'hold_damping':(0,2),'index_load':(.05,.6),'grip':(.05,1.5),'speed':(.02,.8),'pusher_offset':(-.018,.018),'spin_gain':(0,.1),'tangent_kp':(.1,1000),'tangent_kd':(0,20),'palm_shift_x':(-.02,.02)}
OPTIONAL_ARGUMENTS={'tangent_kp','tangent_kd','palm_shift_x'}
OPERATIONS={'cartesian':'fingertip_path','object':'rigid_object_twist','hom':'relative_contact'}

class Policy(GraphPolicy):
 def __init__(self,path):
  super().__init__(path)
  if self.graph['abstraction']not in OPERATIONS:raise ValueError('Unsupported hand abstraction')
  for node in self.graph['nodes'].values():
   if 'terminal'in node:continue
   command=node.get('command')
   if command is None:continue # Historical frozen phase-only graphs.
   if command['operation']!=OPERATIONS[self.graph['abstraction']]:raise ValueError('Command/interface mismatch')
   for k,v in command['arguments'].items():
    if k not in LIMITS or not LIMITS[k][0]<=v<=LIMITS[k][1]:raise ValueError('Invalid bounded hand command: '+k)
   if command.get('contact_mode','spin')not in ['spin','face','driven_spin']:raise ValueError('Unknown contact mode')

def build(interface='hom',goal_deg=90,edge=.02,axis='z',adapter=None,bundle_name=None):
 if interface not in ['cartesian','object','hom']:raise ValueError(interface)
 path=OUT/'policies'/(bundle_name or f'cube{edge*1000:g}-rotate-{axis}-{goal_deg:g}-{interface}');path.mkdir(parents=True,exist_ok=True)
 nodes={
 'acquire':dict(action='acquire',guard='grasp_ready',next='lift'),
 'lift':dict(action='lift',guard='lift_ready',next='rotate'),
 'rotate':dict(action='rotate',guard='near_goal',next='hold'),
 'hold':dict(action='hold',guard='accepted',next='succeeded'),
 'succeeded':dict(terminal='succeeded'),'failed':dict(terminal='failed')}
 contract=dict(edge_m=edge,target_deg=goal_deg,axis=f'initial world {axis} axis, evaluated relative to palm',orientation_error_deg=5,minimum_lift_mm=60,maximum_palm_turn_deg=2,maximum_object_speed_deg_s=5,minimum_side_load_N=.05,hold_s=.3,initial_state='clear fingers; free cube on physical pedestal')
 adapter=adapter or dict(grip=.6,speed=.6,pusher_offset=0.,spin_gain=.06,contact_mode='spin')
 adapter.setdefault('index_load',.15);adapter.setdefault('hold_damping',0.)
 for node in nodes.values():
  if 'action'in node:node['command']=dict(operation=OPERATIONS[interface],arguments={k:adapter[k]for k in LIMITS if k in adapter},contact_mode=adapter['contact_mode'])
 graph=dict(format='dexagent.cube-policy/2',task='cube_rotation',goal=f'Pick up the {edge*1000:g} mm cube, lift it, rotate it {goal_deg:g} degrees in hand about the initial world {axis} axis, and hold.',abstraction=interface,start='acquire',nodes=nodes,contract=contract,graph_hz=20,physics_edit_access=False,adapter_handoff=adapter.get('handoff','continuous'),hold_mode=adapter.get('hold_mode','track'),hold_reset=adapter.get('hold_reset',False))
 code='"""Generated graph nodes: observations in, typed hand command out."""\n'
 for name,n in nodes.items():
  code+=f'\ndef {name}(o):\n'
  if 'terminal'in n:code+=f'    return {name!r}\n'
  else:code+=f'    if o["failure"]: return "failed"\n    if o[{n["guard"]!r}]: return {n["next"]!r}\n    return {name!r}\n'
 code+='\nNODES={'+','.join(f'{n!r}:{n}'for n in nodes)+'}\n'
 (path/'graph.json').write_text(json.dumps(graph,indent=2));(path/'policy.py').write_text(code)
 digest=lambda n:hashlib.sha256((path/n).read_bytes()).hexdigest()
 (path/'manifest.json').write_text(json.dumps(dict(author='Codex in this conversation',runtime_model_invocation=False,source_goal=graph['goal'],video_goal_source=None,code_sha256=digest('policy.py'),graph_sha256=digest('graph.json')),indent=2))
 return path
