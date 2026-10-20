import importlib.util, json, os, tempfile, unittest
from copy import deepcopy
from pathlib import Path
from screwdriver_lab.agentic.contracts import validate
from screwdriver_lab.agentic.runtime import GraphPolicy
from screwdriver_lab.agentic.workflow import compile_policy, digest, generate

ROOT=Path(__file__).resolve().parents[1]/'artifacts/agentic_workflow'


class WorkflowTests(unittest.TestCase):
    def setUp(self):self.graph=json.loads((ROOT/'hom-candidate.json').read_text())
    def test_wrong_abstraction_cannot_bypass_adapter(self):
        self.graph['nodes']['turn']['action']='object_twist'
        with self.assertRaises(ValueError):validate(self.graph)
    def test_no_physics_edits_in_action_arguments(self):
        self.graph['nodes']['turn']['arguments']['friction']=0
        with self.assertRaises(ValueError):validate(self.graph)
    def test_no_silent_full_task_initialization(self):
        self.graph['initial_condition']='horizontal_pickup'
        with self.assertRaisesRegex(ValueError,'no implicit initialization jump'):validate(self.graph)
    def test_nonfinite_commands_rejected(self):
        self.graph['nodes']['turn']['arguments']['speed_deg_s']=float('nan')
        with self.assertRaises(ValueError):validate(self.graph)
    def test_dangling_transition_rejected(self):
        self.graph['nodes']['turn']['otherwise']='unknown'
        with self.assertRaises(ValueError):validate(self.graph)
    def test_failure_route_required(self):
        self.graph['nodes']['failed']['terminal']='succeeded'
        with self.assertRaises(ValueError):validate(self.graph)
    def test_generated_function_names_cannot_shadow_router(self):
        for name in ['route','class','NODES']:
            graph=deepcopy(self.graph);graph['nodes'][name]={'terminal':'failed'}
            with self.assertRaises(ValueError):validate(graph)
    def test_repair_cannot_relax_physical_goal(self):
        from screwdriver_lab.agentic.loop import check_goal
        expected=deepcopy(self.graph['goal']);self.graph['goal']['tolerance']=10
        with self.assertRaisesRegex(ValueError,'fixed physical goal'):check_goal(self.graph,expected)
    def test_goal_hash_is_required(self):
        self.graph['request_sha256']='wrong'
        with tempfile.TemporaryDirectory()as tmp:
            with self.assertRaisesRegex(ValueError,'bound'):compile_policy(self.graph,ROOT/'goal-turn30/request.json',Path(tmp)/'compiled',{})
    def test_generated_code_controls_transitions(self):
        with tempfile.TemporaryDirectory()as tmp:
            dest=Path(tmp)/'compiled';compile_policy(self.graph,ROOT/'goal-turn30/request.json',dest,{'author':'test'})
            spec=importlib.util.spec_from_file_location('test_policy',dest/'policy.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            policy=GraphPolicy(self.graph,module.NODES)
            obs=dict(elapsed_s=0,contact_ready=False,screw_deg=0,depth_mm=0,tool_tilt_deg=0,contact_lost=False,unexpected_force_N=0,seat_contact=False,settled_s=0)
            self.assertEqual(policy.step(obs)['action'],'acquire')
            obs.update(elapsed_s=.3,contact_ready=True);self.assertEqual(policy.step(obs)['action'],'relative_contact')
            obs.update(elapsed_s=1.5,screw_deg=31);self.assertEqual(policy.step(obs)['action'],'hold')
            obs.update(elapsed_s=1.8,settled_s=.3);self.assertEqual(policy.step(obs)['terminal'],'succeeded')
    def test_contact_failure_precedes_target_guard(self):
        p=GraphPolicy(self.graph);p.node='turn'
        obs=dict(elapsed_s=1,contact_lost=True,screw_deg=35,unexpected_force_N=0,tool_tilt_deg=0)
        self.assertEqual(p.step(obs)['terminal'],'failed')
    def test_external_generator_protocol_and_gpu1(self):
        import sys,shlex
        with tempfile.TemporaryDirectory()as tmp:
            script=Path(tmp)/'generator.py'
            script.write_text('import os,json,sys\np=json.load(sys.stdin)\nassert os.environ["CUDA_VISIBLE_DEVICES"]=="1"\nassert len(p["image_inputs"])==7\nprint(json.dumps({"request_sha256":p["request_sha256"]}))\n')
            result=generate(ROOT/'goal-turn30/request.json',shlex.join([sys.executable,str(script)]))
            self.assertEqual(result['request_sha256'],digest(ROOT/'goal-turn30/request.json'))


if __name__=='__main__':unittest.main()
