"""Independently check branch continuity and the recorded seating hold."""
import hashlib
import json
import mujoco
import numpy as np
from screwdriver_lab.hom.model import OUT
from screwdriver_lab.hom.seating import geometry, flush


def validate(name='hom-seat-stop42'):
    report = json.loads((OUT / (name + '.json')).read_text())
    branch = report['checkpoint']
    data = np.load(OUT / (name + '.npz'))
    parent_path = OUT / (branch['source'] + '.npz')
    parent = np.load(parent_path)
    assert hashlib.sha256(parent_path.read_bytes()).hexdigest() == branch['source_npz_sha256']
    for key, end in [('qpos', branch['source_frame'] + 1),
                     ('qvel', branch['source_frame'] + 1),
                     ('time', branch['source_frame'] + 1),
                     ('samples', branch['source_sample'] + 1)]:
        assert np.array_equal(data[key][:end], parent[key][:end]), key
    assert max(branch['continuity'].values()) < 1e-10
    assert hashlib.sha256((OUT / (name + '.xml')).read_bytes()).hexdigest() == report['xml_sha256']
    for filename, digest in report['source_sha256'].items():
        source = OUT / 'source_snapshots' / (digest + '_' + filename)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == digest
    samples = data['samples']
    speed = np.r_[0, np.diff(samples[:, 2]) / np.diff(samples[:, 0])]
    accepted = samples[:, 0] > samples[-1, 0] - .5 + 1e-8
    assert accepted.sum() == 250
    assert np.all(samples[accepted, -2] > .2)
    assert np.all(abs(speed[accepted]) < 5)
    assert np.all(samples[accepted, 13] < 3)
    assert np.all(samples[accepted, 4:7] > [.05, .1, .05])
    model = mujoco.MjModel.from_xml_path(str(OUT / (name + '.xml')))
    state = mujoco.MjData(model)
    for pose, time in zip(data['qpos'], data['time']):
        if time > samples[-1, 0] - .5:
            state.qpos[:] = pose
            mujoco.mj_kinematics(model, state)
            assert flush(geometry(model, state))
    result = dict(id=name, prefix_bitwise_identical=True,
                  raw_parent_hash_verified=True, source_hashes_verified=True,
                  continuous_force_speed_load_criteria_verified=True,
                  all_final_50Hz_poses_flush=True,
                  checkpoint_time_s=branch['time_s'], full_task_success=False)
    (OUT / 'checkpoint-validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == '__main__':
    validate()
