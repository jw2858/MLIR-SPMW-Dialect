import json
import networkx as nx
from build_graph import construct_build_graph

build_graph = construct_build_graph(
    'ninja.txt',
    cc='/home/jw2858/miniconda3/envs/circt/bin/cc',
    cxx='/home/jw2858/miniconda3/envs/circt/bin/c++',
    ar='/home/jw2858/miniconda3/envs/circt/bin/llvm-ar'
)

print('nodes:', len(build_graph.nodes), 'edges:', len(build_graph.edges))

with open('build-graph.json', 'w') as f:
    json.dump(
        nx.node_link_data(build_graph),
        f,
        indent=2,
    )

