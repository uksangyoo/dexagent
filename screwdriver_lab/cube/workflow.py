"""Language request -> validated agent-authored graph -> executable cube policy.

The authoring agent supplies a candidate. This module does not impersonate an
LLM. Physical scene and acceptance criteria remain outside command arguments.
"""
import argparse,hashlib,json,keyword,math
from pathlib import Path
from .policy import LIMITS,OPERATIONS,OPTIONAL_ARGUMENTS

GUARDS={'grasp_ready','lift_ready','near_goal','accepted'}
ACTIONS={'acquire','lift','rotate','hold'}
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def prepare(goal,output,axis='y',degrees=30,edge=.04):
 path=Path(output);path.mkdir(parents=True,exist_ok=False)
 if axis not in ['x','y','z'] or not 0<degrees<=90 or edge not in [.02,.04]:raise ValueError('Unsupported goal contract')
 request=dict(format='dexagent.cube-request/1',goal_prompt=goal,modalities=['language'],contract=dict(edge_m=edge,target_deg=degrees,axis=f'initial world {axis} axis, evaluated relative to palm',orientation_error_deg=5,minimum_lift_mm=60,maximum_palm_turn_deg=2,maximum_object_speed_deg_s=5,minimum_side_load_N=.05,hold_s=.3,initial_state='clear fingers; free cube on physical pedestal'),abstractions=OPERATIONS,command_bounds=LIMITS,optional_arguments=sorted(OPTIONAL_ARGUMENTS),command_units=dict(tangent_kp='N/m motor impedance; does not change contact material',tangent_kd='N s/m motor damping'),observations=sorted(GUARDS|{'failure'}),instructions='Return a graph with request_sha256, abstraction, start, nodes. Nonterminal nodes carry action, guard, next and command {operation, arguments, contact_mode}. Failure routes to failed. Include succeeded and failed terminals. No physics, object pose or force edits. Every success is independently checked from measurements.')
 (path/'request.json').write_text(json.dumps(request,indent=2));return path/'request.json'

def compile_policy(candidate,request_path,output,author):
 request=json.loads(Path(request_path).read_text())
 if candidate.get('request_sha256')!=digest(request_path):raise ValueError('Wrong goal request')
 interface=candidate['abstraction'];nodes=candidate['nodes']
 if interface not in OPERATIONS or not 2<=len(nodes)<=32 or candidate['start']not in nodes:raise ValueError('Invalid graph')
 if nodes.get('succeeded')!={'terminal':'succeeded'} or nodes.get('failed')!={'terminal':'failed'}:raise ValueError('Missing explicit terminals')
 for name,node in nodes.items():
  if not name.isidentifier()or keyword.iskeyword(name)or name=='NODES':raise ValueError('Invalid node identifier')
  if 'terminal'in node:
   if name not in ['succeeded','failed']:raise ValueError('Unknown terminal')
   continue
  if set(node)!={'action','guard','next','command'}or node['action']not in ACTIONS or node['guard']not in GUARDS or node['next']not in nodes:raise ValueError('Invalid node operation or edge')
  c=node['command']
  if set(c)!={'operation','arguments','contact_mode'}or c['operation']!=OPERATIONS[interface]or c['contact_mode']not in ['spin','face','driven_spin']:raise ValueError('Invalid hand command')
  if set(c['arguments'])-set(LIMITS) or (set(LIMITS)-OPTIONAL_ARGUMENTS)-set(c['arguments']):raise ValueError('Unknown or missing hand argument; physics edits forbidden')
  for k,v in c['arguments'].items():
   if isinstance(v,bool)or not isinstance(v,(float,int))or not math.isfinite(v)or not LIMITS[k][0]<=v<=LIMITS[k][1]:raise ValueError('Command out of bounds')
 handoff=candidate.get('adapter_handoff','continuous')
 if handoff not in ['continuous','adapter_reset']:raise ValueError('Unknown adapter handoff')
 hold_mode=candidate.get('hold_mode','track')
 if hold_mode not in ['track','freeze']:raise ValueError('Unknown hold mode')
 hold_reset=candidate.get('hold_reset',False)
 if not isinstance(hold_reset,bool):raise ValueError('Invalid reference-reset option')
 graph=dict(hold_reset=hold_reset,hold_mode=hold_mode,adapter_handoff=handoff,format='dexagent.cube-policy/2',task='cube_rotation',goal=request['goal_prompt'],contract=request['contract'],abstraction=interface,start=candidate['start'],nodes=nodes,graph_hz=20,physics_edit_access=False,request_sha256=digest(request_path))
 code='"""Compiled observation-conditioned agent graph; no physics access."""\n'
 for name,n in nodes.items():
  code+=f'\ndef {name}(o):\n'
  if 'terminal'in n:code+=f'    return {name!r}\n'
  else:code+=f'    if o["failure"]: return "failed"\n    if o[{n["guard"]!r}]: return {n["next"]!r}\n    return {name!r}\n'
 code+='\nNODES={'+','.join(f'{n!r}:{n}'for n in nodes)+'}\n'
 out=Path(output);out.mkdir(parents=True,exist_ok=False);(out/'graph.json').write_text(json.dumps(graph,indent=2));(out/'policy.py').write_text(code)
 (out/'manifest.json').write_text(json.dumps(dict(author=author,runtime_model_invocation=False,source_goal=graph['goal'],request_sha256=digest(request_path),code_sha256=digest(out/'policy.py'),graph_sha256=digest(out/'graph.json')),indent=2));return out

def feedback(result_path,output):
 """Return measured failure evidence to the authoring agent, without retuning physics."""
 import numpy as np
 path=Path(result_path);r=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));s=a['samples']
 pose=(s[:,1]<=5)&(s[:,3]<2)&(s[:,4]>=60)&(np.minimum(s[:,5],s[:,7])>.05)&(s[:,9]<.1)
 packet=dict(format='dexagent.cube-feedback/1',parent_policy_sha256=r['provenance']['graph_sha256'],
  result_sha256=digest(path),recording_sha256=digest(path.with_suffix('.npz')),
  full_task_success=r['success'],prepared_start=bool(r.get('checkpoint')),outcome=r['outcome'],
  goal_contract=r['graph']['contract'],final=r['final'],transitions=r['transitions'],
  diagnostics=dict(maximum_continuous_accepted_hold_s=float(max(s[:,10].max(),r['final']['hold_s'])),
   near_goal_loaded_samples=int(pose.sum()),near_goal_speed_p95_deg_s=float(np.percentile(s[pose,8],95))if pose.any()else None,
   peak_unexpected_load_N=float(max(s[:,9].max(),r['final']['unexpected_N'])),
   peak_motor_effort_Nm=np.max(abs(s[:,11:37]),axis=0).tolist()),
  previous_graph=r['graph'],requested_revision='Revise bounded graph commands or handoff using these observations. Keep the same goal, initial-condition set, contact model and motor limits. A prepared-state success receives no pickup credit.',
  forbidden_changes=['object state assignment','external object wrench','contact parameter changes','goal relaxation'])
 Path(output).write_text(json.dumps(packet,indent=2));return packet

if __name__=='__main__':
 p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
 q=sub.add_parser('prepare');q.add_argument('--goal',required=True);q.add_argument('--axis',choices=['x','y','z'],default='y');q.add_argument('--degrees',type=float,default=30);q.add_argument('--edge',type=float,default=.04);q.add_argument('--output',required=True)
 q=sub.add_parser('compile');q.add_argument('--candidate',required=True);q.add_argument('--request',required=True);q.add_argument('--output',required=True);q.add_argument('--author',required=True)
 q=sub.add_parser('feedback');q.add_argument('--result',required=True);q.add_argument('--output',required=True)
 a=p.parse_args()
 if a.action=='prepare':print(prepare(a.goal,a.output,a.axis,a.degrees,a.edge))
 elif a.action=='compile':print(compile_policy(json.loads(Path(a.candidate).read_text()),a.request,a.output,a.author))
 else:feedback(a.result,a.output)
