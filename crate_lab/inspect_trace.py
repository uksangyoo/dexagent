"""Render recorded physics states without modifying the running simulation."""
from pathlib import Path
import json
import numpy as np
import imageio.v3 as iio
import newton


def keyframes(sim,report="latest.json",camera="inspection"):
    here=Path(__file__).resolve().parent
    data=json.loads((here/"results"/report).read_text())
    rows=data["trace"]
    indices=[0]+[i for i in range(1,len(rows)) if rows[i]["phase"]!=rows[i-1]["phase"]]+list(range(max(0,len(rows)-100),len(rows),20))+[len(rows)-1]
    output=here/"results"/(Path(report).stem+"-"+camera)
    output.mkdir(exist_ok=True)
    if camera=="inspection":
        from robosimstudio.specs import CameraSpec
        cam=CameraSpec(name="inspection",width=1280,height=720,fov_y=45,eye=(2.6,1.5,1.8),look_at=(1.3,0,1.0))
    else:cam=next(c for c in sim.cameras if c.name==camera)
    renderer=sim.make_renderer("cycles",cameras=[cam])
    state=sim.model.state()
    try:
        for i in sorted(set(indices)):
            state.joint_q.assign(np.asarray(rows[i]["joint_q"],dtype=np.float32))
            newton.eval_fk(sim.model,state.joint_q,state.joint_qd,state)
            rgb=renderer.render(state)[cam.name]["rgb"]
            iio.imwrite(output/f'{i:04}_{rows[i]["phase"]}.png',rgb)
            print(f"KEYFRAME {i} {rows[i]['t']} {rows[i]['phase']}",flush=True)
            yield
    finally:renderer.close()
