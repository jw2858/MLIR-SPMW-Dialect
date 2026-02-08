import json
import networkx as nx
from build_graph import parse_cxx_command, yield_command_args

cc = '/home/jw2858/miniconda3/envs/circt/bin/cc'
cxx = '/home/jw2858/miniconda3/envs/circt/bin/c++'

includes_to_outputs = {}

for args in yield_command_args('ninja.txt'):
    if args and args[0] in (cc, cxx):
        cxx_command = parse_cxx_command(args)
        output = cxx_command.output
        includes = tuple(sorted(cxx_command.includes) + sorted(cxx_command.include_systems))
        includes_to_outputs.setdefault(includes, set()).add(output)

sorted_includes = sorted(includes_to_outputs)
sorted_includes_and_sorted_outputs = []
for includes in sorted_includes:
    outputs = includes_to_outputs[includes]
    sorted_outputs = sorted(outputs)
    sorted_includes_and_sorted_outputs.append((list(includes), sorted_outputs))

with open('includes-and-outputs.json', 'w') as f:
    json.dump(sorted_includes_and_sorted_outputs, f, indent=2)
