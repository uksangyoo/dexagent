"""What the model sees each turn: one or more camera images, proprioception and contact forces.

Deliberately absent: `object_pose`, `object_in_palm`, `object_velocity` and
`object_joint`. Where an object is has to be read off the images. (The scorer in
each task's check.py still reads true state, and the `contacts` primitives map
object-frame points through the true pose internally, as they always have.)
"""
import base64
import io

import numpy as np


def _r(a, n=4):
    return np.round(np.asarray(a, float), n).tolist()


def parse_cameras(value):
    """'closeup' or 'closeup,overview' -> ('closeup', 'overview')."""
    if value is None or value == "":
        return ()
    if isinstance(value, (list, tuple)):
        return tuple(str(c).strip() for c in value if str(c).strip())
    return tuple(c.strip() for c in str(value).split(",") if c.strip())


def rgb_array(sim, camera):
    return np.asarray(sim.render(camera)["rgb"])


def rgb_data_url(sim, camera, quality=85):
    """A JPEG of `camera` as a data URL, ready for an image_url content part."""
    import imageio.v3 as iio
    buf = io.BytesIO()
    iio.imwrite(buf, rgb_array(sim, camera), extension=".jpg", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def camera_info(sim, camera):
    """Eye, look-at and vertical field of view of a camera, so the image can be localised in the world."""
    try:
        spec = sim._camera_for(camera)
        return {"name": camera, "eye": _r(spec.eye, 3), "look_at": _r(spec.look_at, 3), "fov_y_deg": float(spec.fov_y),
                "size_px": [int(spec.width), int(spec.height)]}
    except Exception:
        return {"name": camera}


def proprio(api):
    """Time, palm pose (commanded and measured), pad centres and, where the level has it, the 20 finger joints."""
    xyz_c, R_c = api.palm_pose()
    xyz_m, R_m = api.measured_palm()
    out = {
        "t_s": round(api.t, 2),
        "palm_cmd_world": {"xyz_m": _r(xyz_c, 4), "fingers_dir": _r(R_c[:, 2], 3), "palm_normal_dir": _r(R_c[:, 0], 3),
                           "thumb_side_dir": _r(R_c[:, 1], 3)},
        "palm_measured_world": {"xyz_m": _r(xyz_m, 4), "fingers_dir": _r(R_m[:, 2], 3), "palm_normal_dir": _r(R_m[:, 0], 3),
                                "thumb_side_dir": _r(R_m[:, 1], 3)},
        "pad_centres_palm_frame_m": {n: _r(p, 4) for n, p in zip(("thumb", "index", "middle", "ring", "little"), api.tips())},
        "finger_joints_rad": _r(api._q(), 3),
    }
    return out


def forces(api):
    """Contact force [N] between each finger (and the palm) and each scored object: touch, not pose."""
    names = ("thumb", "index", "middle", "ring", "little", "palm")
    out = {}
    for obj in api.objects:
        try:
            out[obj] = {n: round(float(f), 2) for n, f in zip(names, api.contact_forces(obj))}
        except Exception as e:  # noqa: BLE001 - a missing contact table must not end the run
            out[obj] = f"unavailable: {e}"
    return out


def observe(api, cameras):
    """Text observation dict and a list of (camera_name, jpeg_data_url) for every named camera."""
    cams = parse_cameras(cameras)
    if not cams:
        raise ValueError("observe() needs at least one camera name")
    infos = [camera_info(api._sim, c) for c in cams]
    images = [(c, rgb_data_url(api._sim, c)) for c in cams]
    obs = {"proprioception": proprio(api), "contact_forces_N": forces(api), "cameras": infos}
    return obs, images


def side_by_side(rgbs):
    """Concatenate RGB frames left-to-right, matching the shortest height (and cropping width as-is)."""
    frames = [np.asarray(f) for f in rgbs]
    h = min(f.shape[0] for f in frames)
    cropped = []
    for f in frames:
        if f.shape[0] != h:
            y0 = (f.shape[0] - h) // 2
            f = f[y0:y0 + h]
        cropped.append(f)
    return np.concatenate(cropped, axis=1)
