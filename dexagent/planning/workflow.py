"""Goal/video -> bounded pickup/drive graph -> executable observation nodes.

The authoring agent supplies the candidate. Compilation cannot edit the plant,
initial state, success oracle, or motor limits. Feedback retains failed trials.
"""
import argparse, hashlib, json, keyword, math
from pathlib import Path
from dexagent.planning.policy import INTERFACES

LIMITS={'drive_load':(.05,1.3),'return_load':(.005,.2),'speed':(5,90)}
OPTIONAL_LIMITS={'reserve_margin':(0,.2),'tangent_kp':(10,1000),'tangent_kd':(0,20),'alignment_filter_s':(0,.2)}
OBSERVATIONS={'loaded_s','lift_ready','release_ready','tilt_deg','lift_accepted','pivot_accepted','assembly_accepted'}
ACTIONS={'pinch','lift','hold_horizontal','release','pivot','cap_support','assemble'}

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def validate_command(command,interface):
 if set(command)!={'operation','arguments'}or command['operation']!=INTERFACES[interface]['command']:raise ValueError('Invalid hand abstraction command')
 args=command['arguments']
 if set(args)-set(LIMITS)-set(OPTIONAL_LIMITS)-{'cycle'}or (set(LIMITS)|{'cycle'})-set(args):raise ValueError('Unknown hand argument; physics edits forbidden')
 if args['cycle']not in ['paired','alternating']:raise ValueError('Unknown contact cycle')
 for key,(lo,hi)in {**LIMITS,**{k:v for k,v in OPTIONAL_LIMITS.items()if k in args}}.items():
  v=args[key]
  if isinstance(v,bool)or not isinstance(v,(float,int))or not math.isfinite(v)or not lo<=v<=hi:raise ValueError('Command outside motor-control bounds')
 if args['return_load']>=args['drive_load']:raise ValueError('Sliding reset requires lower commanded pressure')
 return args.copy()

def prepare(goal,video,destination):
 out=Path(destination);out.mkdir(parents=True,exist_ok=False);video=Path(video).resolve()
 request=dict(format='dexagent.sequence-request/1',goal_prompt=goal,modalities=['language','video'],video=dict(path=str(video),sha256=digest(video)),task='pickup_pivot_drive',abstractions=INTERFACES,observations=sorted(OBSERVATIONS|{'failure'}),command_bounds=LIMITS,optional_command_bounds=OPTIONAL_LIMITS,command_units=dict(drive_load='N',return_load='N',speed='degree/s',reserve_margin='rad',tangent_kp='N/m',tangent_kd='N s/m',alignment_filter_s='s'),cycles=['paired','alternating'],contract=dict(continuous_pickup_required=True,prepared_state_credit=False,measured_thread_advance_required=True,seated_loaded_hold_s=.5,goal_relaxation_allowed=False),instructions='Return observation-conditioned graph nodes. Assembly may use bounded pressure, stroke speed and contact-cycle commands. All actuation goes through 20 hand plus 6 arm motors. Do not change geometry, contact properties, body state, object forces, or success criteria.')
 (out/'request.json').write_text(json.dumps(request,indent=2));return out/'request.json'

def compile_policy(candidate,request_path,destination,author):
 request=json.loads(Path(request_path).read_text())
 if candidate.get('request_sha256')!=digest(request_path):raise ValueError('Wrong goal request')
 interface=candidate['abstraction'];nodes=candidate['nodes']
 if interface not in INTERFACES or not 2<=len(nodes)<=32 or candidate['start']not in nodes:raise ValueError('Invalid graph')
 if nodes.get('succeeded')!={'terminal':'succeeded'}or nodes.get('failed')!={'terminal':'failed'}:raise ValueError('Missing terminal nodes')
 for name,node in nodes.items():
  if not name.isidentifier()or keyword.iskeyword(name)or name=='NODES':raise ValueError('Invalid node name')
  if 'terminal'in node:
   if name not in ['succeeded','failed']:raise ValueError('Unknown terminal')
   continue
  if set(node)-{'action','edges','otherwise','command'}or node.get('action')not in ACTIONS or node.get('otherwise')not in nodes:raise ValueError('Invalid operation')
  if 'command'in node:
   if node['action']!='assemble':raise ValueError('Drive command outside assembly')
   validate_command(node['command'],interface)
  for key,op,value,target in node['edges']:
   if key not in OBSERVATIONS or op not in ['gt','lt','eq']or target not in nodes or not isinstance(value,(bool,int,float))or not math.isfinite(value):raise ValueError('Invalid observed edge')
 graph=dict(format='dexagent.pickup-policy/2',task=request['task'],goal=request['goal_prompt'],contract=request['contract'],abstraction=interface,interface=INTERFACES[interface],start=candidate['start'],nodes=nodes,graph_hz=20,physics_edit_access=False,request_sha256=digest(request_path))
 code='"""Compiled graph: observations and bounded hand commands only."""\n'
 for name,node in nodes.items():
  code+=f'\ndef {name}(o):\n'
  if 'terminal'in node:code+=f'    return {name!r}\n';continue
  code+='    if o["failure"]: return "failed"\n'
  for key,op,value,target in node['edges']:
   operator={'gt':'>','lt':'<','eq':'=='}[op];code+=f'    if o[{key!r}] {operator} {value!r}: return {target!r}\n'
  code+=f'    return {node["otherwise"]!r}\n'
 code+='\nNODES={'+','.join(f'{n!r}:{n}'for n in nodes)+'}\n'
 out=Path(destination);out.mkdir(parents=True,exist_ok=False);(out/'graph.json').write_text(json.dumps(graph,indent=2));(out/'policy.py').write_text(code)
 (out/'manifest.json').write_text(json.dumps(dict(author=author,runtime_model_invocation=False,request_sha256=digest(request_path),video_goal_source=request['video']['path'],video_sha256=request['video']['sha256'],code_sha256=digest(out/'policy.py'),graph_sha256=digest(out/'graph.json')),indent=2));return out

def feedback(result_path,destination):
 import numpy as np
 path=Path(result_path);r=json.loads(path.read_text());s=np.load(path.with_suffix('.npz'))['samples']
 packet=dict(format='dexagent.sequence-feedback/1',result_sha256=digest(path),recording_sha256=digest(path.with_suffix('.npz')),outcome=r['outcome'],full_task_success=r['success'],checkpoint=r.get('checkpoint'),observed=r['assembly_final'],assembly_transitions=r['assembly_trace'],previous_parameters=r['parameters'],maximum_screw_turn_deg=float(s[:,2].max()),peak_motor_effort_Nm=np.max(abs(s[:,14:40]),axis=0).tolist(),requested_revision='Revise graph commands using measured failure, then repeat the complete sequence. Keep physics, motor limits, task and initial-condition set fixed.')
 Path(destination).write_text(json.dumps(packet,indent=2));return packet

if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 q=sub.add_parser('prepare');q.add_argument('--goal',required=True);q.add_argument('--video',required=True);q.add_argument('--output',required=True)
 q=sub.add_parser('compile');q.add_argument('--candidate',required=True);q.add_argument('--request',required=True);q.add_argument('--output',required=True);q.add_argument('--author',required=True)
 q=sub.add_parser('feedback');q.add_argument('--result',required=True);q.add_argument('--output',required=True)
 a=p.parse_args()
 if a.action=='prepare':print(prepare(a.goal,a.video,a.output))
 elif a.action=='compile':print(compile_policy(json.loads(Path(a.candidate).read_text()),a.request,a.output,a.author))
 else:feedback(a.result,a.output)
