import json
import os
import posixpath
import sys
from collections import OrderedDict
import networkx as nx


with open('relpath-build-graph.json', 'r') as f:
    node_link_data = json.load(f)
    relpath_build_graph = nx.node_link_graph(node_link_data)

o_files = []
for file in relpath_build_graph.nodes:
    _, ext = os.path.splitext(file)
    if ext == '.o':
        o_files.append(file)
o_files.sort()

o_files_to_source_extensions_to_sources = OrderedDict()
for o_file in o_files:
    source_extensions_to_sources = {}
    for source, _ in relpath_build_graph.in_edges(o_file):
        _, source_ext = os.path.splitext(source)
        source_extensions_to_sources.setdefault(source_ext, []).append(source)
    for sources in source_extensions_to_sources.values():
        sources.sort()
    o_files_to_source_extensions_to_sources[o_file] = source_extensions_to_sources

with open('o-file-sources.json', 'w') as f:
    json.dump(o_files_to_source_extensions_to_sources, f, indent=2)

