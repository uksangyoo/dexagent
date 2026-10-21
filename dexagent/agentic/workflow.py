"""Prepare a goal/video request, accept agent output, and emit executable code.

An external generator is a configured program receiving the request JSON on
stdin and returning graph JSON on stdout. No canned policy is substituted for
an unavailable language model. This conversation's example is explicitly
attributed to the authoring agent, not to an independent runtime API call.
"""
import argparse, hashlib, json, os, shlex, subprocess
from pathlib import Path
from dexagent.agentic.contracts import ABSTRACTIONS, FORMAT, OBSERVATIONS, validate


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(goal, destination, video=None):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    request=dict(format='dexagent.goal-request/1',goal_prompt=goal,
                 modalities=['language'],gpu_policy=dict(physical_gpu=1,physics='CPU Drake'),
                 abstractions=ABSTRACTIONS,
                 output_contract=dict(format=FORMAT,required=['request_sha256','abstraction','initial_condition','goal','start','nodes'],initial_condition='prepared_grasp',goal={'metric':'screw_rotation_deg','target':'1..720','tolerance':'0.1..10','hold_s':'0.1..2'},node={'action':'acquire | hold | chosen abstraction operation','arguments':'optional speed_deg_s 5..90 and stroke_deg 5..30','edges':[{'observation':sorted(OBSERVATIONS),'operator':'ge | le | eq','value':'finite number or bool','to':'node id'}],'otherwise':'node id'},terminal={'terminal':'succeeded | failed'}),
                 execution_capabilities={'prepared_grasp_rotation':True,'full_pickup_to_seat':False},
                 instructions='Produce a JSON policy graph using dexagent.goal-policy/1. Select one hand abstraction. Use observation-conditioned nodes and bounded commands. Do not assign object motion, modify contact physics, or claim pickup success from a prepared grasp. Unsupported full-task stages must be reported as missing capabilities, not silently removed.')
    if video:
        path=Path(video).resolve();request['modalities'].append('video');request['video']=dict(path=str(path),sha256=digest(path),frames=[])
        for t in [0,1,2,3,4,6,8]:
            frame=destination/f'frame-{t:02d}.jpg'
            subprocess.run(['ffmpeg','-v','error','-ss',str(t),'-i',str(path),'-frames:v','1','-vf','scale=640:-2','-y',str(frame)],check=True)
            if frame.exists():request['video']['frames'].append(dict(time_s=t,path=frame.name,sha256=digest(frame)))
    (destination/'request.json').write_text(json.dumps(request,indent=2))
    return request


def compile_policy(candidate, request, destination, provenance):
    validate(candidate)
    if candidate.get('request_sha256') != digest(request): raise ValueError('Policy is not bound to this goal request')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    graph_text=json.dumps(candidate,indent=2);(destination/'graph.json').write_text(graph_text)
    # Code nodes are generated from the validated typed graph, with no eval or
    # execution of arbitrary model-produced Python. Each function is executable.
    code='"""Generated policy code; consumes observations and returns typed hand commands."""\nfrom dexagent.agentic.runtime import route\n\n'
    for name,node in candidate['nodes'].items():
        code+=f'def {name}(observation):\n    node = {node!r}\n'
        code+="    return {'node': node, 'next': "+(repr(name) if 'terminal' in node else 'route(node, observation)')+"}\n\n"
    code+='NODES = {'+', '.join(repr(n)+': '+n for n in candidate['nodes'])+'}\n'
    (destination/'policy.py').write_text(code)
    manifest=dict(request_sha256=digest(request),graph_sha256=digest(destination/'graph.json'),code_sha256=digest(destination/'policy.py'),provenance=provenance,graph_format=FORMAT,compatible_with_gap_harness=False,physical_gpu=1)
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2));return manifest


def generate(request, command, repair=None):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':'1'}
    payload={'request_sha256':digest(request),'request_directory':str(Path(request).resolve().parent),'request':json.loads(Path(request).read_text())}
    payload['image_inputs']=[str(Path(request).resolve().parent/f['path'])for f in payload['request'].get('video',{}).get('frames',[])]
    if repair is not None: payload['execution_feedback']=repair
    result=subprocess.run(shlex.split(command),input=json.dumps(payload),text=True,capture_output=True,env=env,check=True,timeout=300)
    return json.loads(result.stdout)


def feedback(result, destination):
    r=json.loads(Path(result).read_text())
    packet=dict(format='dexagent.repair-request/1',parent_policy_sha256=r['policy_sha256'],
                outcome=r['outcome'],observed=r['final'],transitions=r['agent_transitions'],
                requested_revision='Revise policy/graph or abstraction commands using observed failure; keep physics and actuator limits fixed.',
                forbidden_changes=['object state assignment','external object wrench','friction changes','success-threshold relaxation'])
    Path(destination).write_text(json.dumps(packet,indent=2));return packet


if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='action',required=True)
    q=sub.add_parser('prepare');q.add_argument('--goal',required=True);q.add_argument('--video');q.add_argument('--output',required=True)
    q=sub.add_parser('compile');q.add_argument('--candidate',required=True);q.add_argument('--request',required=True);q.add_argument('--output',required=True);q.add_argument('--author',required=True)
    q=sub.add_parser('generate');q.add_argument('--request',required=True);q.add_argument('--command',required=True);q.add_argument('--output',required=True)
    q=sub.add_parser('feedback');q.add_argument('--result',required=True);q.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='prepare':prepare(a.goal,a.output,a.video)
    elif a.action=='compile':compile_policy(json.loads(Path(a.candidate).read_text()),a.request,a.output,{'author':a.author,'mode':'explicit_agent_artifact'})
    elif a.action=='generate':compile_policy(generate(a.request,a.command),a.request,a.output,{'mode':'external_generator','command_program':shlex.split(a.command)[0]})
    else:feedback(a.result,a.output)
