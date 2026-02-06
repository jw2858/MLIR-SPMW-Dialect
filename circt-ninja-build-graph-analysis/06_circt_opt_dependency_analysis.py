import json
import os
import posixpath
import sys
import networkx as nx


def networkx_reverse_bfs_layers(G, sources):
    """Yield layers of nodes in reverse BFS order (using predecessors).

    Parameters
    ----------
    G : NetworkX graph (DiGraph)
    sources : iterable of nodes

    Yields
    ------
    layer : list
        The current layer of nodes
    """
    visited = set()
    # Use set to avoid duplicate nodes in a layer
    current_layer = set(sources)
    while current_layer:
        yield list(current_layer)
        visited.update(current_layer)
        next_layer = set()
        for node in current_layer:
            for pred in G.predecessors(node):
                if pred not in visited:
                    next_layer.add(pred)
        current_layer = next_layer


with open('relpath-build-graph.json', 'r') as f:
    node_link_data = json.load(f)
    relpath_build_graph = nx.node_link_graph(node_link_data)

all_layers_extensions_to_files = []
for layer in networkx_reverse_bfs_layers(relpath_build_graph, ['build/bin/circt-opt']):
    layer.sort()
    extensions_to_files = {}
    for file in layer:
        _, ext = os.path.splitext(file)
        extensions_to_files.setdefault(ext, []).append(file)
    all_layers_extensions_to_files.append(extensions_to_files)

with open('circt-opt-dependencies.json', 'w') as f:
    json.dump(all_layers_extensions_to_files, f, indent=2)
