"""Executable observation-conditioned graph, without the GaP harness.

This uses the general nodes / conditional-edges structure of graph-as-policy.
It is a small local runner, not a claim of compatibility with GaP's v3 schema.
Each node is actual Python policy code; transitions consume measured contact
loads. The low-level adapter and the physics backend are separate modules.
"""
import numpy as np


def acquire(c, dt):
    if c.cycle == 'alternating':
        return 'loaded' if c.ready > .1 else 'wait'
    return 'loaded' if c.h.d.time-c.phase_start > .25 and c.ready > .1 else 'wait'


def drive(c, dt):
    return 'stroke_complete' if c.stroke >= c.amplitude else 'wait'


def unload(c, dt):
    light = min(c.filtered[[0, 2]]) > .003 and max(c.filtered[[0, 2]]) < c.return_load*1.8
    return 'light_contact' if c.h.d.time-c.phase_start > .1 and light and c.filtered[1] > .3 else 'wait'


def reset(c, dt):
    if c.stroke <= 0:
        c.cycles += 1
        return 'reset_complete'
    return 'wait'


def hold(c, dt):
    return 'wait'


def alternating(c, dt):
    c.driver = 0 if c.phase == 'thumb_drive' else 1
    forces = c.filtered[[0, 2]]
    if forces[c.driver] > .7*c.drive_load and .02 < forces[1-c.driver] < c.return_load*1.8 and c.filtered[1] > .2:
        c.strokes[c.driver] = min(c.amplitude, c.strokes[c.driver]+c.speed*dt)
        c.strokes[1-c.driver] = max(0, c.strokes[1-c.driver]-c.speed*dt)
    if c.strokes[c.driver] >= c.amplitude and c.strokes[1-c.driver] <= 0:
        c.cycles += .5
        return 'stroke_complete'
    return 'wait'


FUNCTIONS = dict(acquire=acquire, drive=drive, unload=unload, reset=reset,
                 thumb_drive=alternating, middle_drive=alternating, hold=hold)


def specification(cycle, interface):
    edges = {'acquire': {'loaded': 'thumb_drive' if cycle == 'alternating' else 'drive'}}
    if cycle == 'alternating':
        edges.update(thumb_drive={'stroke_complete': 'middle_drive'}, middle_drive={'stroke_complete': 'thumb_drive'})
    else:
        edges.update(drive={'stroke_complete': 'unload'}, unload={'light_contact': 'reset'}, reset={'reset_complete': 'acquire'})
    for mapping in edges.values():mapping['seated']='hold'
    edges['hold']={}
    return dict(format='graph-as-policy-inspired/local-executable-1',
                meta=dict(interface=interface, scope='prepared-pose manipulation', runtime_llm=False),
                start='acquire', nodes={n: dict(type='code', ref=__name__+'.'+FUNCTIONS[n].__name__) for n in edges},
                conditional_edges={n: dict(router_field='exit', mapping={**m, 'wait': n}) for n, m in edges.items()})


def advance(c, dt):
    stable = min(c.filtered[[0, 2]]) > .7*c.drive_load and max(c.filtered[[0, 2]]) < 1.3*c.drive_load and c.filtered[1] > .2
    c.ready = c.ready+dt if stable else 0.
    old = c.strokes.copy()
    if getattr(c,'seating_requested',False) and c.phase!='hold':c.transition('hold')
    # One node per tick; acquire can hand off immediately to its first stroke.
    name = c.phase
    event = FUNCTIONS[name](c, dt)
    target = c.graph['conditional_edges'][name]['mapping'][event]
    if target != name:
        c.transition(target)
        if name == 'acquire' and c.cycle == 'alternating':
            FUNCTIONS[target](c, dt)
    if c.phase=='hold':
        c.omegas[:]=0
        load=getattr(c,'hold_load',c.drive_load)
        c.normal_targets+=(1-np.exp(-dt/getattr(c,'load_ramp_s',.15)))*(load-c.normal_targets)
        return load, 0.
    if c.cycle == 'alternating':
        goal = np.full(2, c.drive_load)
        if c.phase != 'acquire':
            c.driver = 0 if c.phase == 'thumb_drive' else 1
            goal[:] = c.return_load
            goal[c.driver] = c.drive_load
        c.normal_targets += (1-np.exp(-dt/getattr(c,'load_ramp_s',.15)))*(goal-c.normal_targets)
        c.omegas = (c.strokes-old)/dt
        return c.drive_load, 0.
    moving = c.phase == 'drive' or (c.phase == 'reset' and min(c.filtered[[0, 2]]) > .003)
    omega = c.speed if c.phase == 'drive' else -c.speed if moving else 0.
    c.stroke = np.clip(c.stroke+omega*dt, 0, c.amplitude)
    goal=c.return_load if c.phase in ['unload', 'reset'] else c.drive_load
    c.normal_targets += (1-np.exp(-dt/getattr(c,'load_ramp_s',.15)))*(goal-c.normal_targets)
    return float(c.normal_targets.mean()), omega


from pathlib import Path as _SourcePath
__source_bytes__ = _SourcePath(__file__).read_bytes()
