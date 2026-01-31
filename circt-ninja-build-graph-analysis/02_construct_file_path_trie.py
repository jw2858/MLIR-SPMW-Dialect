import json
import networkx as nx
import pickle
import posixpath
from fspathverbs import Root, Parent, Current, Child, compile_to_fspathverbs
from tinytrie import TrieNode, update

def to_path_components(verbs):
    path_components = []
    for verb in verbs:
        if isinstance(verb, Root):
            path_components = [verb.root]
        elif isinstance(verb, Parent):
            path_components.pop()
        elif isinstance(verb, Current):
            pass
        elif isinstance(verb, Child):
            path_components.append(verb.child)
    return path_components


with open('build-graph.json', 'r') as f:
    node_link_data = json.load(f)
    build_graph = nx.node_link_graph(node_link_data)

root = TrieNode()

for node in build_graph.nodes:
    verbs = compile_to_fspathverbs(node, posixpath.split)
    path_components = to_path_components(verbs)
    update(root, path_components, node)

with open('file-path-trie.pkl', 'wb') as f:
    pickle.dump(root, f)
