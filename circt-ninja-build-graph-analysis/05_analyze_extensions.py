import json
import posixpath
from collections import Counter
import networkx as nx


with open('relpath-build-graph.json', 'r') as f:
    node_link_data = json.load(f)
    relpath_build_graph = nx.node_link_graph(node_link_data)

target_extensions_to_targets_to_sources = {}

for target in relpath_build_graph.nodes:
    in_edges = relpath_build_graph.in_edges(target)
    if in_edges:
        _, target_ext = posixpath.splitext(target)
        for source, _ in in_edges:
            target_extensions_to_targets_to_sources.setdefault(target_ext, {}).setdefault(target, []).append(source)

with open('target-extensions-to-targets-to-sources.json', 'w') as f:
    json.dump(target_extensions_to_targets_to_sources, f, indent=2)
