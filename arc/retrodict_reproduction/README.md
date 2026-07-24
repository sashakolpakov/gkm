# Taint-controlled Retrodict reproduction

Retrodict's published `containment.json` only checks that `arcengine` and
`arc_agi` cannot be imported from its analysis virtual environment. The analysis
subprocess otherwise inherits the host filesystem, environment, and network.

For a controlled reproduction, the Python analysis tool is instead run in the
image defined by `Dockerfile.analysis` with:

- only the run workspace bind-mounted at `/workspace`;
- `--network none`;
- a read-only container root;
- no host environment variables;
- dropped Linux capabilities and `no-new-privileges`;
- PID, memory, CPU, and temporary-storage limits.

The game engine and downloaded `environment_files` remain in the host runner and
are never mounted into the analysis container. ThinHarness's native file tools
are separately rooted to the workspace.

The reproduction is invalidated if the retained trace contains an attempted
reference to game packages/source, a path outside the workspace, credentials,
process execution, or networking. A blocked attempt still counts as taint for
the purpose of comparing the agent's behavior, even though the container keeps
it from succeeding.

