# Maintained diagnostics

This directory contains small, reusable repository checks that do not belong in
the runtime package. Diagnostic scripts must be deterministic. They must accept
inputs at run time and must not use credentials, change the Hub, patch caches,
or contain machine-specific paths.

`check_notation.py` checks the notation rules for documentation and comments.
The release test suite also runs it.

Put one-off parity investigations in an untracked work directory. Add an
investigation here only when it becomes a supported diagnostic with tests and
documented inputs.
