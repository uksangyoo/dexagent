"""Run a declared numerical sweep sequentially on physical GPU 1."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'screwdriver_lab/artifacts/physics_validation'
PY = ROOT/'screwdriver_lab/vendor/contact-env/bin/python'
NEWTON = ROOT/'third-party/newton'
CASES = [
    ['--mode','native'], ['--mode','point'],
    ['--mode','hydro','--resolution','32'],
    ['--mode','hydro','--resolution','64'],
    ['--mode','hydro','--resolution','128'],
    ['--mode','hydro','--resolution','64','--dt','.0005'],
    ['--mode','hydro','--resolution','128','--dt','.0005'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--mass','.05'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--mass','.2'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--kh','1e9'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--kh','1e10'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--platen-ratio','100'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--gap','.0005'],
    ['--mode','hydro','--resolution','64','--dt','.0005','--loading','twist'],
    ['--mode','point','--dt','.0005','--loading','twist'],
]


if __name__ == '__main__':
    BASE.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OPENBLAS_NUM_THREADS='1', PYTHONPATH=str(NEWTON))
    manifest = []
    for i, args in enumerate(CASES):
        log = BASE/f'upstream-case-{i:02}.log'
        command = [str(PY), '-u', '-m', 'screwdriver_lab.contact_bench', *args]
        print('RUN', i, args, flush=True)
        with log.open('w') as f:
            result = subprocess.run(command, env=env, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        manifest.append(dict(case=i, command=command, exit_code=result.returncode, log=log.name))
        (BASE/'sweep-manifest.json').write_text(json.dumps(manifest,indent=2))
        print('END', i, result.returncode, flush=True)
        if result.returncode:
            print(log.read_text()[-2000:], flush=True)
