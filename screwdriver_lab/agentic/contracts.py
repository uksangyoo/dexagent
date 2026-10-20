"""The agent's typed command surface. Physics is outside this interface."""
import math, keyword

FORMAT = 'dexagent.goal-policy/1'
ABSTRACTIONS = {
    'cartesian': {'operation': 'finger_paths', 'meaning': 'World-frame fingertip arcs; path position is the controlled quantity.'},
    'object': {'operation': 'object_twist', 'meaning': 'Rigid tool twist mapped to fingertip translation and orientation.'},
    'hom': {'operation': 'relative_contact', 'meaning': 'Object rotation plus relative sliding; two rolling coordinates released, contact-normal spin constrained.'},
}
OBSERVATIONS = {'contact_ready', 'screw_deg', 'depth_mm', 'tool_tilt_deg', 'contact_lost',
                'unexpected_force_N', 'seat_contact', 'settled_s', 'elapsed_s', 'node_elapsed_s'}
OPERATORS = {'ge', 'le', 'eq'}


def validate(graph):
    if graph.get('format') != FORMAT: raise ValueError('Unsupported graph format')
    interface = graph.get('abstraction')
    if interface not in ABSTRACTIONS: raise ValueError('Unknown hand abstraction')
    if graph.get('initial_condition') != 'prepared_grasp':
        raise ValueError('This runner supports prepared_grasp only. Pickup/transport have not passed physical validation; no implicit initialization jump is allowed.')
    goal = graph['goal']
    if goal.get('metric') != 'screw_rotation_deg': raise ValueError('Unsupported physical goal metric')
    for name, lower, upper in [('target', 1, 720), ('tolerance', .1, 10), ('hold_s', .1, 2)]:
        value = goal[name]
        if not isinstance(value, (float, int)) or isinstance(value,bool) or not math.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f'Invalid goal {name}')
    nodes = graph['nodes']
    if not isinstance(nodes,dict) or not 1 <= len(nodes) <= 32 or graph['start'] not in nodes: raise ValueError('Invalid graph nodes/start')
    allowed = {'acquire', 'hold', ABSTRACTIONS[interface]['operation']}
    for name, node in nodes.items():
        if not name.isidentifier() or keyword.iskeyword(name) or name in {'route','NODES'}: raise ValueError('Node IDs must be nonreserved Python identifiers')
        if 'terminal' in node:
            if node['terminal'] not in ['succeeded','failed']: raise ValueError('Invalid terminal outcome')
            continue
        if node.get('action') not in allowed: raise ValueError(f'Unsupported operation: {node.get("action")}')
        args = node.get('arguments', {})
        if set(args) - {'speed_deg_s','stroke_deg'}: raise ValueError('Unknown command argument; agents cannot edit physics or object state')
        for key, val in args.items():
            bounds = (5, 90) if key == 'speed_deg_s' else (5, 30)
            if not isinstance(val,(int,float)) or isinstance(val,bool) or not math.isfinite(val) or not bounds[0] <= val <= bounds[1]: raise ValueError('Command outside declared bounds')
        if node.get('otherwise') not in nodes: raise ValueError('Missing otherwise edge')
        for edge in node.get('edges',[]):
            if edge.get('to') not in nodes: raise ValueError('Dangling edge')
            if edge.get('observation') not in OBSERVATIONS or edge.get('operator') not in OPERATORS: raise ValueError('Unknown observation guard')
            if not isinstance(edge.get('value'),(int,float,bool)) or not math.isfinite(edge['value']): raise ValueError('Invalid guard threshold')
    reachable, stack = set(), [graph['start']]
    while stack:
        name=stack.pop()
        if name in reachable: continue
        reachable.add(name);node=nodes[name]
        if 'terminal' not in node: stack += [node['otherwise']] + [e['to']for e in node.get('edges',[])]
    if not any(nodes[n].get('terminal') == 'failed' for n in reachable): raise ValueError('Graph needs a reachable failure outcome')
    if not any(nodes[n].get('terminal') == 'succeeded' for n in reachable): raise ValueError('Graph needs a reachable success outcome')
    return graph
