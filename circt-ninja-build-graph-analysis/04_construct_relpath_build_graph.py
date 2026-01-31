import json
import posixpath
import networkx as nx


with open('build-graph.json', 'r') as f:
    node_link_data = json.load(f)
    build_graph = nx.node_link_graph(node_link_data)

with open('paths-to-relpaths.json', 'r') as f:
    paths_to_relpaths = json.load(f)

relpath_build_graph = nx.DiGraph()

for u, v in build_graph.edges:
    if u in paths_to_relpaths:
        relpath_u = paths_to_relpaths[u]
    else:
        continue

    if v in paths_to_relpaths:
        relpath_v = paths_to_relpaths[v]
    else:
        continue

    relpath_build_graph.add_edge(relpath_u, relpath_v)

with open('relpath-build-graph.json', 'w') as f:
    node_link_data = nx.node_link_data(relpath_build_graph)
    json.dump(node_link_data, f, indent=2)
