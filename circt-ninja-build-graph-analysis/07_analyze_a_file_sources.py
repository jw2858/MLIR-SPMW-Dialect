import json
import os
import posixpath
import sys
from collections import OrderedDict
import networkx as nx


with open('relpath-build-graph.json', 'r') as f:
    node_link_data = json.load(f)
    relpath_build_graph = nx.node_link_graph(node_link_data)

a_files = []
for file in relpath_build_graph.nodes:
    _, ext = os.path.splitext(file)
    if ext == '.a':
        a_files.append(file)
a_files.sort()

a_files_to_source_extensions_to_sources = OrderedDict()
for a_file in a_files:
    source_extensions_to_sources = {}
    for source, _ in relpath_build_graph.in_edges(a_file):
        _, source_ext = os.path.splitext(source)
        source_extensions_to_sources.setdefault(source_ext, []).append(source)
    for sources in source_extensions_to_sources.values():
        sources.sort()
    a_files_to_source_extensions_to_sources[a_file] = source_extensions_to_sources

with open('a-file-sources.json', 'w') as f:
    json.dump(a_files_to_source_extensions_to_sources, f, indent=2)

