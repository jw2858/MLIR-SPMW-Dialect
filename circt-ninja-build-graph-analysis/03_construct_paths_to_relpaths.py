import json
import pickle
import posixpath
import tinytrie

with open('file-path-trie.pkl', 'rb') as f:
    root = pickle.load(f)

paths_to_relpaths = {}

for subtrie_name, subtrie_root in root.children.items():
    if subtrie_name != '/':
        for components, node in tinytrie.collect_sequences(subtrie_root):
            path = node.value
            relpath = posixpath.join('build', subtrie_name, *components)
            paths_to_relpaths[path] = relpath

project_subtrie_prefix = ['/', 'home', 'jw2858', 'circt-main']

project_subtrie_root = tinytrie.get_subtrie_root(root, project_subtrie_prefix)

for components, node in tinytrie.collect_sequences(project_subtrie_root):
    path = node.value
    relpath = posixpath.join(*components)
    paths_to_relpaths[path] = relpath

with open('paths-to-relpaths.json', 'w') as f:
    json.dump(paths_to_relpaths, f, indent=2)
