"""Configured authoring agent → typed code → dynamics → feedback → revision.

No provider is silently substituted. Each attempt has a new immutable bundle
and rollout. An independently supplied goal contract prevents the authoring
agent from changing the physical acceptance criteria between attempts.
"""
import argparse, json, os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='1'
from .workflow import prepare, generate, compile_policy, feedback, digest
from .contracts import validate


def check_goal(candidate, expected):
    validate(candidate)
    if candidate['goal'] != expected:
        raise ValueError('Agent output changes the fixed physical goal contract')
    return candidate


def execute(request, goal_contract, command, directory, max_attempts=3, duration=4):
    from .run import run
    from screwdriver_lab.hom.model import OUT
    if not 1<=max_attempts<=10:raise ValueError('Use 1..10 bounded attempts')
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    expected=json.loads(Path(goal_contract).read_text());repair=None;attempts=[]
    # Validation exceptions stop the loop; they are not mistaken for a physical failure.
    for i in range(max_attempts):
        candidate=check_goal(generate(request,command,repair),expected)
        bundle=directory/f'policy-{i+1:02d}'
        compile_policy(candidate,request,bundle,dict(mode='external_generator',attempt=i+1,parent_policy_sha256=repair['parent_policy_sha256']if repair else None))
        name=f'{directory.name}-attempt-{i+1:02d}'
        result=run(bundle,name,duration)
        repair=feedback(OUT/(name+'.json'),directory/f'feedback-{i+1:02d}.json')
        repair['previous_graph']=candidate
        attempts.append(dict(policy=str(bundle),result=str(OUT/(name+'.json')),outcome=result['outcome']))
        (directory/'attempts.json').write_text(json.dumps(dict(goal=expected,request_sha256=digest(request),attempts=attempts,physical_gpu=1),indent=2))
        if result['success']:break
    return attempts


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--goal-contract',required=True);p.add_argument('--generator-command',required=True);p.add_argument('--output',required=True);p.add_argument('--max-attempts',type=int,default=3);p.add_argument('--duration',type=float,default=4);a=p.parse_args()
    execute(a.request,a.goal_contract,a.generator_command,a.output,a.max_attempts,a.duration)
