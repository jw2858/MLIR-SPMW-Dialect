Extracts and analyzes build dependency graph from logs produced by the Ninja build system when building [CIRCT](https://github.com/llvm/circt/tree/main/).

Main capabilities include:

- **Parsing Ninja output** to extract individual tool commands.
- **Tokenizing and decoding command-line invocations** for compilers (`cc`, `c++`) and `ar`.
- **Constructing a build dependency graph** (`networkx.DiGraph`) where nodes represent files and edges show build dependencies.
- **Mapping absolute file paths to build-relative paths** for easier downstream usage.
- **Extension-based analysis** (e.g., finding all `.a` files and their sources).
- **Intermediate JSON files** for reproducibility.

## Python Script Descriptions

### `01_construct_build_graph.py`

Reads a raw Ninja build log (`ninja.txt`), parses out all C/C++ and archive build steps, and creates a directed build dependency graph using `networkx`. Saves the graph in JSON format (`build-graph.json`).

### `02_construct_file_path_trie.py`

Parses the build graph to extract all unique file paths from build nodes. Decomposes each path into components and constructs a prefix trie (using `tinytrie`), serializing it to disk as `file-path-trie.pkl`. This trie structure allows for fast prefix/path lookups.

### `03_construct_paths_to_relpaths.py`

Loads the file path trie, walks the trie, and generates a mapping from absolute build artifact paths to project-relative paths (`paths-to-relpaths.json`). Ensures files are mapped both for external and project-tree artifacts.

### `04_construct_relpath_build_graph.py`

Converts the build graph's absolute paths into their project-relative equivalents using the mapping from the previous step, creating a cleaner graph (`relpath-build-graph.json`) with edges between relative file paths.

### `05_analyze_extensions.py`

Analyzes the relpath build graph to group output files by their extension (e.g., `.o`, `.a`, binaries). For each extension, creates a mapping from targets to their source inputs, and writes this analysis to `target-extensions-to-targets-to-sources.json`.

### `06_circt_opt_dependency_analysis.py`

Performs a dependency layer analysis of the main CIRCT binary (`build/bin/circt-opt`). Walks backward from the `circt-opt` binary in the build graph `relpath-build-graph.json` using reverse BFS, producing `circt-opt-dependencies.json`, extension breakdowns for each dependency layer. Useful for tracking how each component of the binary is built up from its source dependencies.

### `07_analyze_a_file_sources.py`

Finds all static library (`.a`) files in the build graph `relpath-build-graph.json`. For each archive, reports the file extensions and lists all sources (by extension) that were used to assemble the archive to `a-file-sources.json`. Useful for analyzing how each `.a` file is composed in the build.

### `08_analyze_o_file_sources.py`

Finds all object (`.o`) files in the build graph `relpath-build-graph.json`. For each object file, reports the file extensions and lists all sources (by extension) that were used to produce that object to `o-file-sources.json`. Enables fine-grained analysis of object-file-level build provenance.
