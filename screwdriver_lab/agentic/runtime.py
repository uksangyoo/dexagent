"""Small GaP-inspired observation router; no LLM runs in the motor loop."""
from .contracts import validate


def route(node, observation):
    for edge in node.get('edges', []):
        value=observation[edge['observation']]; threshold=edge['value']
        matched={'ge':lambda:value>=threshold,'le':lambda:value<=threshold,'eq':lambda:value==threshold}[edge['operator']]()
        if matched: return edge['to']
    return node['otherwise']


class GraphPolicy:
    def __init__(self, graph, code_nodes=None):
        self.graph=validate(graph);self.node=graph['start'];self.entered=0.;self.trace=[];self.code_nodes=code_nodes

    def step(self, observation):
        observation={**observation,'node_elapsed_s':observation['elapsed_s']-self.entered}
        node=self.graph['nodes'][self.node]
        if 'terminal' not in node:
            target=self.code_nodes[self.node](observation)['next'] if self.code_nodes else route(node,observation)
            if target != self.node:
                self.trace.append(dict(t=observation['elapsed_s'],from_node=self.node,to=target,observation=observation))
                self.node=target;self.entered=observation['elapsed_s'];node=self.graph['nodes'][target]
        return {'node':self.node,**node}
